# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, Dataset
import string
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForTriviaQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    RLQABertForQuestionAnswering,
    BertForWikihopMulti,
    AUTOQABertForWikihopMulti
    
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import json
# from triviaqa_utils import evaluation_utils

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import shutil
logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForWikihopMulti, BertTokenizer),
    "autobert": (BertConfig, AUTOQABertForWikihopMulti, BertTokenizer),
}

TEST_SPEED = False

class WikihopDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_candidates, ignore_seq_with_no_answers, max_question_len, max_segment, train_rl=False, race_npy_path=None,
                 train_teacher=False):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            self.data_json = json.load(f)
            print('done reading file')

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_candidates = max_num_candidates
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        self.max_segment = max_segment
        self.zero_item = [0] * self.max_seq_len
        self.one_zero_item = [1] + [0] * (self.max_seq_len-1)

        self.train_rl = train_rl
        if train_rl:
            file_name = "npy_folder/wikihop_"+str( self.max_seq_len )+"_"+str( self.max_segment )+".memmap"   
            self.guide_file = np.memmap(filename = file_name, shape=(len(self.data_json),  self.max_segment, 2, self.max_seq_len), mode='r', dtype=np.float32)       

        self.train_teacher = train_teacher
        if self.train_teacher:
            self.teacher_logits = np.load('npy_folder/wikihop_'+str( self.max_seq_len )+'_logits.npy')


    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)

    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        qid = example['id']
        question_text = example['query']
        candidates = example['candidates']
        supports = example['supports']
        answer = example['answer']


        # q_start = '[unused1]'
        # q_end = '[unused2]'
        cand_start = '[unused1]'
        context_start = '[unused2]'

        query_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(question_text) 

        heads = query_tokens
        cand_positions = []
        answer_index = -1
        for cand in candidates:
            if cand==answer:
                answer_index = len(cand_positions)
            cand_positions.append(len(heads))

            heads += [cand_start] + self.tokenizer.tokenize(cand) 

        assert( answer_index >= 0 )
        padding_len = self.max_num_candidates - len(cand_positions)
        cand_positions.extend([-1] * padding_len)

        all_doc_tokens = []            
        for support in supports:
            all_doc_tokens += [context_start] + self.tokenizer.tokenize(support)

        all_doc_tokens = all_doc_tokens[:self.max_doc_len]

        max_tokens_per_doc_slice = self.max_seq_len - len(heads) - 1
        assert max_tokens_per_doc_slice > 0
        if self.doc_stride < 0:
            # negative doc_stride indicates no sliding window, but using first slice
            self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
        
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        instance_mask = []
        # cand_positions_list = []
        cnt = 0
        for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens =  heads + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(heads)) + [1] * (len(doc_slice_tokens) + 1)
            assert len(segment_ids) == len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_len = self.max_seq_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len



            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            instance_mask.append(1)
            # cand_positions_list.append(cand_positions)

            cnt += 1
            if cnt >= self.max_segment:
                break
        

        while cnt < self.max_segment:
            input_ids_list.append(self.zero_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            instance_mask.append(0)
            cnt += 1


        item = [torch.tensor(input_ids_list), 
                                torch.tensor(input_mask_list),
                                torch.tensor(segment_ids_list),
                                torch.tensor(cand_positions), 
                                torch.tensor(instance_mask),
                                torch.tensor(answer_index)#.view(-1)
                                ]
        if self.train_rl:
            item.append( torch.from_numpy(self.guide_file[idx]) )

        if self.train_teacher:
            item.append( torch.from_numpy(self.teacher_logits[idx]) )

        return item


    @staticmethod
    def collate_one_doc_and_lists(batch):
        # num_metadata_fields = 0
        fields = [x for x in zip(*batch)]
        # stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        # stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        stacked_fields = [torch.stack(field) for field in fields] 
        global TEST_SPEED
        if TEST_SPEED:
            vaild = torch.sum(stacked_fields[4], dim=0)
            for j in range(stacked_fields[0].shape[1])[::-1]:
                if vaild[j]!=0:
                    break
            stacked_fields[0] = stacked_fields[0][:,0: j+1,:]
            stacked_fields[1] = stacked_fields[1][:,0: j+1,:]
            stacked_fields[2] = stacked_fields[2][:,0: j+1,:]
            stacked_fields[4] = stacked_fields[4][:,0: j+1]
    
        return stacked_fields


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def to_list(tensor):
    return tensor.detach().cpu().tolist()



def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/wikihop_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment)
                            

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)
                    


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
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
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
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "cand_positions": batch[3],
                "instance_mask": batch[4],
                "answer_index": batch[5],
            }

            # if args.model_type in ["xlnet", "xlm"]:
            #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
             
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
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
                if args.local_rank in [-1, 0]  and (args.save_steps > 0 and (global_step % args.save_steps == 0 or global_step==10)):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('acc:', acc)

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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(args, model, tokenizer, prefix=""):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    global TEST_SPEED
    TEST_SPEED = True
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataset = WikihopDataset(file_path=args.dev_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment)


    eval_sampler = SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    eval_accuracy = 0
    eval_examples = 0
    # tot_loss = 0
    macs_list = json.load(open('macs_list.json'))
    flops = 0
    bert_flops = 0
    bert_512_flops = 0

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()

        batch = [t.to(args.device) for t in batch]
        max_segment = batch[1].shape[1]
        L_s = batch[1].view(-1, batch[1].shape[-1]).sum(dim=1)
        l = int(torch.max(L_s))
        batch[0] = batch[0][:, :, :l]
        batch[1] = batch[1][:, :, :l]
        batch[2] = batch[2][:, :, :l]
        
        with torch.no_grad():

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "cand_positions": batch[3],
                "instance_mask": batch[4],
                # "answer_index": batch[5],
            }

            outputs = model(**inputs)
            logits = outputs[0]


        logits = to_list(logits)
        label_ids = to_list(batch[5])

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_examples += len(logits)

        hidden_size = 768#model.hidden_size if hasattr(model, "hidden_size") else model.module.hidden_size
        num_labels = 1#model.num_labels if hasattr(model, "num_labels") else model.module.num_labels

        if args.model_type.startswith('auto'):

            all_lens = outputs[1]
            ci = len(all_lens)
            lens = []
            for j in range(ci):
                r, _ = torch.max(all_lens[j].view(batch[0].size(0), max_segment), dim=-1)
                lens.append(r)

            times = [1, 5, 6] # (for 1,6)

            while (len(lens)+1 <len(times)):
                rt = times[-1]
                times = times[:-1]
                times[-1] += rt
            for i in range(batch[0].size(0)):

                flops += macs_list[l] * times[0] * max_segment 
                for j in range(len(lens)):
                    lj = int(lens[j][i]*l+0.5) 
                    flops += macs_list[lj] * times[j+1] * max_segment 
                    flops += lj * (hidden_size*32+32*1) * max_segment


                flops += hidden_size * num_labels * max_segment
        bert_flops +=  macs_list[l] * 12 * max_segment + hidden_size * num_labels * max_segment

     
        
    print ('Flops:', 2*flops / len(dataset) / 1000000.0)
    print ('BERT FLOPS:', 2*bert_flops/len(dataset)/1000000.0)

    acc = eval_accuracy / eval_examples
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    TEST_SPEED = False

    return {'acc': acc, 'Flops': 2*flops / len(dataset) / 1000000.0}
    



def evaluate_grad(args, model, tokenizer, prefix=""):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)

    dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment)

    # Note that DistributedSampler samples randomly    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    eval_accuracy = 0
    eval_examples = 0
    tot_loss = 0
    output_dim = [0, 5]
    max_segment = args.max_segment
    max_seq_length = args.max_seq_len

 
    file_name = "npy_folder/wikihop_"+str( max_seq_length)+"_"+str(max_segment)+".memmap"
    layers_scores = np.memmap(filename = file_name, shape=(len(dataset),  max_segment, len(output_dim), max_seq_length), mode='w+', dtype=np.float32)       
 
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
            "cand_positions": batch[3],
            "instance_mask": batch[4],
            "answer_index": batch[5],
        }

        outputs = model(**inputs)
        loss, logits = outputs[:2]

        hidden_states = outputs[2] 
        hidden_states = hidden_states[1:]

        last_val = hidden_states[8]
        
        grads = torch.autograd.grad(loss, [last_val,])
        grad_delta = grads[0] 
        
        for idx_idx, idx in enumerate(output_dim):                
            with torch.no_grad():
                delta = last_val - hidden_states[idx]
                dot = torch.einsum("bli,bli->bl", [grad_delta, delta])
                score = dot.abs()  # (bsz, seq_len)
                score = score.view(-1, max_segment, max_seq_length).detach()

            score = score.cpu().numpy()
            layers_scores[cnt: cnt+score.shape[0], :, idx_idx, :] = score

        cnt += score.shape[0]

        tot_loss += loss.item()

        logits = to_list(logits)
        label_ids = to_list(batch[5])

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_examples += len(logits)


    acc = eval_accuracy / eval_examples
    print (acc)
    return {'acc': acc}



def evaluate_logits(args, model, tokenizer, prefix=""):


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)

    dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment)

    # Note that DistributedSampler samples randomly    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    eval_accuracy = 0
    eval_examples = 0
    tot_loss = 0
    max_segment = args.max_segment
    max_seq_length = args.max_seq_len

    all_logits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()
        with torch.no_grad():

            batch = tuple(t.to(args.device) for t in batch)
            bsz, max_segment, _ =batch[0].shape
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "cand_positions": batch[3],
                "instance_mask": batch[4],
                # "answer_index": batch[5],
            }

            outputs = model(**inputs)
            logits = outputs[0]

            logits = logits.cpu().numpy()
            all_logits.append(logits)
            


    all_logits = np.concatenate(all_logits, axis=0)
    print (all_logits.shape)
    np.save("npy_folder/wikihop_"+str(max_seq_length)+"_logits.npy", all_logits)

    return {}
    

def train_rl(args, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/wikihop_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                train_rl=args.train_rl,
                                race_npy_path=args.race_npy_path)
                            

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)
                    


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
    if args.output_dir.find('test_one')!=-1:

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
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
    max_segment = args.max_segment
    sample_K = 8
    all_copy_rate = 0.0
    jianshao = all_copy_rate / (t_total*0.2)

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

            copy_rate = torch.ones(bsz*sample_K, max_segment)  * all_copy_rate

            if global_step < args.guide_rate * t_total:
                copy_rate[:bsz] = 1

            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "cand_positions": batch[3],
                "instance_mask": batch[4],
                "answer_index": batch[5],
                "copy_rate": copy_rate,
                "tokens_prob": batch[6],
            }

            

            outputs = model(**inputs)


            loss = outputs[0]
            output_selector_loss = outputs[1]
            all_lens = outputs[3]
            instance_mask = batch[4].to(loss)

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(ci):
                l = torch.sum(all_lens[j].view(L, max_segment), dim=-1) / torch.sum(instance_mask, dim=-1)
                lens.append(l)
              
            
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
                qa_loss = qa_loss / args.gradient_accumulation_steps
                selector_loss = selector_loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += qa_loss.item()
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
                all_copy_rate = max(0, all_copy_rate - jianshao)

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("qa_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss

                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL


                if args.local_rank in [-1, 0] and (args.save_steps > 0 and (global_step % args.save_steps == 0 )):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('acc:', acc)

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


def train_both(args, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/wikihop_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                train_rl=args.train_rl,
                                race_npy_path=args.race_npy_path,
                                train_teacher=args.train_teacher)
                            

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler, # num_workers=1, 
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)
                    


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
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
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
    # max_seq_length = args.max_seq_length
    # steps_trained_in_current_epoch = 0
    # punishment = args.punishment
    # torch.autograd.set_detect_anomaly(True)

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
    max_segment = args.max_segment
    sample_K = 8 # old 5
    all_copy_rate = 0.0
    jianshao = all_copy_rate / (t_total*0.2)

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

            copy_rate = torch.ones(bsz*sample_K, max_segment)  * all_copy_rate

            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "cand_positions": batch[3],
                "instance_mask": batch[4],
                "answer_index": batch[5],
                "copy_rate": copy_rate,
            }

            
            if args.train_teacher:
                inputs['teacher_logits'] = batch[-1].to(args.device)

            outputs = model(**inputs)


            loss = outputs[0]
            output_selector_loss = outputs[1]
            all_lens = outputs[3]
            instance_mask = batch[4].to(loss)

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(ci):
                l = torch.sum(all_lens[j].view(L, max_segment), dim=-1) / torch.sum(instance_mask, dim=-1)
                lens.append(l)
              
            
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
            loss = qa_loss + selector_loss 


        
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) trainin

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                qa_loss = qa_loss / args.gradient_accumulation_steps
                selector_loss = selector_loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += qa_loss.item()
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
                all_copy_rate = max(0, all_copy_rate - jianshao)

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("qa_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss

                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL


                if args.local_rank in [-1, 0] and (args.save_steps > 0 and (global_step % args.save_steps == 0 )):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('acc:', acc)

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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
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
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default="wikihop",
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default="train.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default="dev_sort.json",
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
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

  
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )



    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
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
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--do_eval_grad",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--race_npy_path",
                        default="npy_folder/wikihop_test.npy",
                        type=str,
                        required=False)



    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")


    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum length of seq passed to the transformer model")
    parser.add_argument("--max_doc_len", type=int, default=1024,
                        help="Maximum number of wordpieces of the input document")
    parser.add_argument("--max_num_candidates", type=int, default=79,
                        help="")
    parser.add_argument("--max_question_len", type=int, default=55,
                        help="Maximum length of the question")
    parser.add_argument("--doc_stride", type=int, default=-1,
                        help="Overlap between document chunks. Use -1 to only use the first chunk")
    parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                        help="each example should have at least one answer. Default is False")
    parser.add_argument("--test", action='store_true', help="Test only, no training")
    parser.add_argument("--max_segment", type=int, default=8, help="8 for 512 bert")

    parser.add_argument("--alpha", default=1.0, type=float, help="")
    parser.add_argument("--guide_rate", default=0.5, type=float, help="")


    parser.add_argument("--train_teacher",
                        default=False,
                        action='store_true',
                        help=" ")



    parser.add_argument("--do_eval_logits",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    args = parser.parse_args()

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)


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
        if args.output_dir.endswith("test"):
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
        create_exp_dir(args.output_dir, scripts_to_save=['run_wikihop.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_autoqabert.py'])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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
        if args.train_rl:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, train_pred=True)
            global_step, tr_loss = train_rl(args , model, tokenizer)
        elif args.train_both:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train_both(args, model, tokenizer)
        else:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train(args, model, tokenizer)
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
        # if args.do_train:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        # else:
        #     logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        #     checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)
            print (result)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.do_eval_grad and args.local_rank in [-1, 0]:
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_grad(args, model, tokenizer, prefix=global_step)


    if args.do_eval_logits and args.local_rank in [-1, 0]:
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_logits(args, model, tokenizer, prefix=global_step)




    return results


if __name__ == "__main__":
    main()