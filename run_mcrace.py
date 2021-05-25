# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import argparse
import glob
import logging
import os
import random
import timeit
import json
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    RobertaForMultipleChoice,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    AUTOMCBertForMultipleChoice,
    AUTOMCDistilBertForMultipleChoice,
    # InitBertForRace,
    # InitDistilBertForRace,
    DistilBertForMultipleChoice
)

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import math

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
    (),
)

MODEL_CLASSES = {
    # "initbert": (BertConfig, InitBertForRace, BertTokenizer),
    "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMultipleChoice, DistilBertTokenizer),
    "autobert": (BertConfig, AUTOMCBertForMultipleChoice, BertTokenizer),
    # "distilinitbert": (DistilBertConfig, InitDistilBertForRace, DistilBertTokenizer),
    "distilautobert": (DistilBertConfig, AUTOMCDistilBertForMultipleChoice, DistilBertTokenizer),

}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 question_end=None

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        self.question_end = question_end

## paths is a list containing all paths
def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            context_sentence = article,
                            start_ending = question,
                            ending_0 = options[0],
                            ending_1 = options[1],
                            ending_2 = options[2],
                            ending_3 = options[3],
                            label = truth))
                
    return examples 



def convert_examples_to_features(examples, tokenizer, max_seq_length, max_qa_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # ending_tokens = ending_tokens[-max_qa_length:]
    # context_tokens_choice = context_tokens_choice[: (max_seq_length - 3 - len(ending_tokens))]

    print (len(examples))
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        max_qa_L = 0
        for ending_index, ending in enumerate(example.endings):
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            max_qa_L = max(max_qa_L, len(ending_tokens))
        max_qa_L = min(max_qa_L, max_qa_length)
        max_context_L = max_seq_length - 3 - max_qa_L

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[ : max_context_L]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            ending_tokens = ending_tokens[ : max_qa_L]
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label,
                question_end = (len(context_tokens_choice) + 2) + len(start_ending_tokens)
            )
        )

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.int)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.bool)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.bool)
    all_label = torch.tensor([f.label for f in features], dtype=torch.int)
    question_ends = torch.tensor([f.question_end for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, question_ends)

    return features, dataset

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/race_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        args.warmup_steps = int(t_total*0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    torch.autograd.set_detect_anomaly(True)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
         
         
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].long(),
                "labels": batch[3].long(),

            }
            if args.model_type in ["xlm", "roberta", "distilbert"]:
                inputs.pop('token_type_ids')

            outputs = model(**inputs)

            loss = outputs[0]
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and (args.save_steps > 0 and global_step % args.save_steps == 0):
                    update = True
                    if args.evaluate_during_training :
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if acc > best_acc:
                            print ('best acc:', acc)
                            best_acc = acc
                        else:
                            update = False
                    if update:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", evaluate_prefix='dev'):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, evaluate=True, output_examples=False)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    middle_eval_accuracy = middle_nb_eval_examples = 0

    
    macs_list = json.load(open('macs_list.json'))
    flops = 0
    bert_flops = 0
    distilbert_flops = 0
    try:
        layer_num = len(model.bert.encoder.layer) if hasattr(model, "bert") else len(model.module.bert.encoder.layer)
        isbert = True
    except:
        layer_num = len(model.distilbert.transformer.layer) if hasattr(model, "distilbert") else len(model.module.distilbert.transformer.layer)
        isbert = False

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = [t.to(args.device) for t in batch]

        L_s = batch[1].view(-1, batch[1].shape[-1]).sum(dim=1)
        l = int(torch.max(L_s))
        batch[0] = batch[0][:, :, :l]
        batch[1] = batch[1][:, :, :l]
        batch[2] = batch[2][:, :, :l]


        with torch.no_grad():

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"] else batch[2].long(),
                "question_ends": batch[4].long()
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"]:
                inputs.pop('token_type_ids')
            outputs = model(**inputs)
            logits = to_list(outputs[0])
            label_ids = to_list(batch[3])
            
            tmp_eval_accuracy = accuracy(logits, label_ids)
            middle_eval_accuracy += tmp_eval_accuracy
            middle_nb_eval_examples += len(logits)

            
            if args.model_type.find('auto')!=1 and args.per_gpu_eval_batch_size==1:
                hidden_size = 768 
                num_labels = model.num_labels if hasattr(model, "num_labels") else model.module.num_labels
                lens = outputs[1]

                times = [1, layer_num//2-1, layer_num//2]  
                while (len(lens)+1 <len(times)):
                    rt = times[-1]
                    times = times[:-1]
                    times[-1] += rt
                for i in range(batch[0].size(0)):

                    flops += macs_list[l] * times[0] * 4 
                    for j in range(len(lens)):
                        lj = int(lens[j][i]*l+0.5) 
                        flops += macs_list[lj] * times[j+1] * 4 
                        flops += lj * (hidden_size*32+32*1) * 4


                    flops += hidden_size * num_labels * 4 + hidden_size*hidden_size*4
                    bert_flops +=  macs_list[l] * layer_num * 4 + hidden_size * num_labels * 4 + hidden_size*hidden_size*4
                    distilbert_flops +=  macs_list[l] * layer_num//2 * 4 + hidden_size * num_labels * 4 + hidden_size*hidden_size*4


    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / (len(dataset)))
    print ('FLOPS:', 2*flops/len(dataset)/1000000.0)
    if isbert:
        print ('BERT FLOPS:', 2*bert_flops/len(dataset)/1000000.0)
        print ('DistilBERT FLOPS:', 2*distilbert_flops/len(dataset)/1000000.0)

    eval_accuracy = middle_eval_accuracy / middle_nb_eval_examples

    results = {'acc': eval_accuracy, 'FLOPS': 2*flops/len(dataset)/1000000.0}
    print (results)
    return results

def load_and_cache_examples(args, tokenizer, prefix, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}".format(
            prefix,
            #list(filter(None, args.model_name_or_path.split("/"))).pop(),
            # args.model_type,
            str(args.max_seq_length),
        ),
    )
    if args.do_lower_case:
        cached_features_file += '_lower'


    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)

        features_and_dataset = torch.load(cached_features_file)
        dataset = features_and_dataset["dataset"]

        tmp  = list(dataset.tensors)
        if len(tmp)==4:
            features = features_and_dataset["features"]
            question_ends = torch.tensor([f.question_end for f in features], dtype=torch.long)
            tmp.append(question_ends)
            dataset = TensorDataset(*tmp)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        data_dir = os.path.join(args.data_dir, prefix)
        examples = read_race_examples([os.path.join(data_dir, 'high'), os.path.join(data_dir, 'middle')])

        features, dataset = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_qa_length=args.max_qa_length, is_training=True)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"dataset": dataset}, cached_features_file)


    if not evaluate:
        if args.model_type.find('distil')!=-1:
            model_type = 'distilbert'
        else:
            model_type = 'bert'

        if args.train_rl:
            tmp  = list(dataset.tensors)

            tokens_prob = np.load(os.path.join("npy_folder", "race_" + model_type + "_"+ str(args.max_seq_length)+ "_score.npy"))
            tokens_prob = torch.from_numpy(tokens_prob)
            print (tokens_prob.shape)
            
            tmp.append(tokens_prob)
            dataset = TensorDataset(*tmp)

        elif args.train_init:
            tmp  = list(dataset.tensors)

            tokens_rank = np.load(os.path.join("npy_folder", "race_" + model_type + "_"+ str(args.max_seq_length)+ "_rank.npy"))
            tokens_rank = torch.from_numpy(tokens_rank)
            print (tokens_rank.shape)

            tmp.append(tokens_rank)

            dataset = TensorDataset(*tmp)

        elif args.train_teacher:

            tmp  = list(dataset.tensors)
            teacher_logits = np.load(args.teacher_npy_path)    
            teacher_logits = torch.from_numpy(teacher_logits)    # num_ins, 4
            print ('load teacher_logits')
            tmp.append(teacher_logits)

            dataset = TensorDataset(*tmp)



    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def train_init(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/race_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # if args.output_dir.find('test')!=-1:
    #     train_sampler = SequentialSampler(train_dataset) 
    # else:
    #     train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs



    train_params = []
    for n, p in model.named_parameters():        
        # if n.find('bert.encoder.linear_1.3')!=-1 or n.find('bert.encoder.linear_2.3')!=-1:
        if n.find('linear_1')!=-1 or n.find('linear_2')!=-1 or n.find('linear_3')!=-1 \
            or n.find('linear_4')!=-1 or n.find('linear_5')!=-1 \
                or n.find('linear_6')!=-1 or n.find('linear_7')!=-1  \
                    or n.find('linear_7')!=-1 or n.find('linear_8')!=-1 :
            train_params.append((n, p))
            print (n)
        else:
            p.requires_grad = False
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        args.warmup_steps = int(t_total*0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0


    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):


            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"] else batch[2].long(),
                "question_ends": batch[4].long(),
                "rank_prob": batch[5]
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"]:
                inputs.pop('token_type_ids')
                
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def train_rl(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/race_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)


    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    train_params = []
    for n, p in model.named_parameters():        
        if n.find('linear_1')!=-1 or n.find('linear_2')!=-1 or n.find('linear_3')!=-1 \
            or n.find('linear_4')!=-1 or n.find('linear_5')!=-1 \
                or n.find('linear_6')!=-1 or n.find('linear_7')!=-1  \
                    or n.find('linear_7')!=-1 or n.find('linear_8')!=-1 :
            train_params.append((n, p))
            print (n)
        else:
            p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        args.warmup_steps = int(t_total*0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_selector_loss, logging_selector_loss = 0.0, 0.0
    tr_dL, logging_dL = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    sample_K = 8

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            bsz = batch[0].size(0)
            if sample_K > 1:
                new_batch = []
                for t in batch:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat( tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch

            batch = tuple(t.to(args.device) for t in batch)

            copy_rate = torch.zeros(bsz*sample_K,)   

            if global_step < args.guide_rate * t_total:
                copy_rate[:bsz] = 1

            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"] else batch[2].long(),
                "labels": batch[3].long(),
                "question_ends": batch[4].long(),
                "tokens_prob": batch[5],
                "copy_rate": copy_rate, 
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"]:
                inputs.pop('token_type_ids')
            outputs = model(**inputs)

            loss = outputs[0]
            output_selector_loss = outputs[1]
            all_lens = outputs[3]

            L = len(loss)
            race_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(reduction_times):
                lens.append(all_lens[j])

            bsz = L // sample_K
            rewards = []
            for i in range(bsz):
                rewards.append([])
            dLs = []


            for i in range(L):
                delta_L = 0
                for j in range(reduction_times):
                    delta_L += lens[j][i].item()

                delta_L /= reduction_times
                reward = -loss[i].item() - args.alpha * delta_L
            
                rewards[i % bsz].append( reward )
                dLs.append( delta_L )


            mean_dl = sum(dLs)/len(dLs)
            tr_dL += mean_dl / args.gradient_accumulation_steps

            selector_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                rs = np.array(rs)
                baseline = np.mean(rs)

                for j in range(sample_K):
                    reward = (rs[j] - baseline) 
                    selector_loss += reward * output_selector_loss[j*bsz+i]
            selector_loss = selector_loss / L
            loss = selector_loss 


        
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) trainin

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                race_loss = race_loss / args.gradient_accumulation_steps
                selector_loss = selector_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += race_loss.item()
            tr_selector_loss += selector_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("race_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss

                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL


                if (args.local_rank in [-1, 0] and (args.save_steps > 0 and (global_step % args.save_steps == 0))):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if acc > best_acc:
                            print ('Best acc:', acc)
                            best_acc = acc
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.output_dir.find('test_one')!=-1:
                print (loss, race_loss)
                break

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step



def train_both(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/race_log/" + args.output_dir[args.output_dir.rfind('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)


    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        args.warmup_steps = int(t_total*0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_selector_loss, logging_selector_loss = 0.0, 0.0
    tr_dL, logging_dL = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    sample_K = 8

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):


            bsz = batch[0].size(0)
            if sample_K > 1:
                new_batch = []
                for t in batch:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat( tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch

            batch = tuple(t.to(args.device) for t in batch)

            copy_rate = torch.zeros(bsz*sample_K, )   
            
            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert" , "distilautobert"] else batch[2].long(),
                "labels": batch[3].long(),
                "question_ends": batch[4].long(),
                "copy_rate": copy_rate, 
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"]:
                inputs.pop('token_type_ids')

            if args.train_teacher:
                inputs['teacher_logits'] = batch[-1]

            outputs = model(**inputs)

            loss = outputs[0]
            output_selector_loss = outputs[1]
            loss_ori = outputs[2]
            all_lens = outputs[-1]

            L = len(loss)
            race_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(reduction_times):
                lens.append(all_lens[j])

            bsz = L // sample_K
            rewards = []
            for i in range(bsz):
                rewards.append([])
            dLs = []
                
            for i in range(L):
                delta_L = 0
                for j in range(reduction_times):
                    delta_L += lens[j][i].item()

                delta_L /= reduction_times
                reward = -loss_ori[i].item() - args.alpha * delta_L
                rewards[i % bsz].append( reward )
                dLs.append( delta_L )


            mean_dl = sum(dLs)/len(dLs)
            tr_dL += mean_dl / args.gradient_accumulation_steps

            selector_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                rs = np.array(rs)
                baseline = np.mean(rs)

                for j in range(sample_K):
                    reward = (rs[j] - baseline) 
                    selector_loss += reward * output_selector_loss[j*bsz+i]
            selector_loss = selector_loss / L
            loss = race_loss + selector_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) trainin

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                race_loss = race_loss / args.gradient_accumulation_steps
                selector_loss = selector_loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


                        
            tr_loss += race_loss.item()
            tr_selector_loss += selector_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("race_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss

                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL


                if (args.local_rank in [-1, 0]  and (args.save_steps > 0 and (global_step % args.save_steps == 0))) and _>=2:
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if acc > best_acc:
                            print ('Best acc:', acc)
                            best_acc = acc
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_grad(args, model, tokenizer, prefix="", evaluate_prefix='dev'):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, output_examples=False)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    start_time = timeit.default_timer()
    max_seq_length = args.max_seq_length

    if args.model_type=='distilbert':
        output_dims = [0, 2]
        back_r = 4
    elif args.model_type=='bert':
        output_dims = [0, 5]
        back_r = 8
    else:
        assert(False)

    layers_scores = np.zeros((len(dataset), len(output_dims), max_seq_length), dtype=np.float32)
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)


        inputs = {
            "input_ids": batch[0].long(),
            "attention_mask": batch[1].long(),
            "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].long(),
            "labels": batch[3].long(),
 
        }
        if args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
        if args.model_type in ["distilbert",  "distilautobert"]:
            inputs.pop("token_type_ids")

        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2] 
        hidden_states = hidden_states[1:]

        last_val = hidden_states[back_r]
        

        grads = torch.autograd.grad(loss, [last_val,])
        grad_delta = grads[0] 
        
        for idx_idx, idx in enumerate(output_dims):                
            with torch.no_grad():
                delta = last_val - hidden_states[idx]
                dot = torch.einsum("bli,bli->bl", [grad_delta, delta])
                score = dot.abs()  # (bsz, seq_len)
                score = score.view(-1, 4, max_seq_length)
                score = torch.sum(score, dim=1)
                score = score.detach()

            score = score.cpu().numpy()
            layers_scores[cnt: cnt+score.shape[0], idx_idx] = score


        cnt += score.shape[0]

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    np.save(os.path.join("npy_folder", "race_" + args.model_type + "_"+ str(max_seq_length)+ "_score.npy"), layers_scores)
    return {}


def evaluate_logits(args, model, tokenizer, prefix="", evaluate_prefix='dev'):
    dataset = load_and_cache_examples(args, tokenizer, prefix=evaluate_prefix, output_examples=False)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer() 
    all_logits = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = list(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert", "distilautobert"] else batch[2].long(),
            }


            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["distilbert",  "distilautobert"]:
                inputs.pop("token_type_ids")

            outputs = model(**inputs)


            logits = outputs[0]
            logits = logits.cpu().numpy()
            all_logits.append(logits)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / (len(dataset)-1))


    all_logits = np.concatenate(all_logits, axis=0)
    print (all_logits.shape)
    np.save("npy_folder/race_logits.npy", all_logits)

    return {}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_qa_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
 
    parser.add_argument("--do_calc",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_eval_grad",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_eval_logits",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training",
                        default=False,
                        action='store_true',
                        help="")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")


    parser.add_argument("--train_init",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--train_teacher",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--alpha", default=1.0, type=float, help="")
    parser.add_argument("--guide_rate", default=0.2, type=float, help="")

    parser.add_argument("--teacher_npy_path",
                        default="npy_folder/race_logits.npy",
                        type=str,
                        required=False)

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.find("test")!=-1:
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_mcrace.py', 'transformers/src/transformers/modeling_autobert.py',
        'transformers/src/transformers/modeling_automcbert.py'])


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.do_eval_grad:
        config.output_hidden_states = True

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        if args.train_both:
            train_dataset = load_and_cache_examples(args, tokenizer, prefix='train', output_examples=False,)
            global_step, tr_loss = train_both(args, train_dataset, model, tokenizer)
        elif args.train_rl:
            train_dataset = load_and_cache_examples(args, tokenizer, prefix='train', output_examples=False)
            global_step, tr_loss = train_rl(args, train_dataset, model, tokenizer)
        elif args.train_init:
            train_dataset = load_and_cache_examples(args, tokenizer, prefix='train', output_examples=False)
            global_step, tr_loss = train_init(args, train_dataset, model, tokenizer)
        else:
            train_dataset = load_and_cache_examples(args, tokenizer, prefix='train', output_examples=False)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)


        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step, evaluate_prefix='test' if args.do_test else 'dev')

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))


    if args.do_eval_grad and args.local_rank in [-1, 0]:

        logger.info("Loading checkpoint %s for evaluation", args.output_dir)
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_grad(args, model, tokenizer, prefix=global_step, evaluate_prefix='train')

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)




    if args.do_eval_logits and args.local_rank in [-1, 0]:

        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]


        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_logits(args, model, tokenizer, prefix=global_step, evaluate_prefix='train')

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)



    return results



if __name__ == "__main__":
    main()

