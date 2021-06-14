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
import random
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnsweringHotpot,
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
    AUTOQABertForQuestionAnswering,
    AUTOQABertForQuestionAnsweringHotpot
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_final_text,
    _compute_softmax,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import collections
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import shutil
from hotpot_evaluate_v1 import f1_score
import pickle

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig,)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnsweringHotpot, BertTokenizer),
    "autobert": (BertConfig, AUTOQABertForQuestionAnsweringHotpot, BertTokenizer),
}


DO_EVAL_GRAD = False
DO_GUIDE  = False


current_ep = None

class HotpotDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, train_rl=False, train_teacher=False):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print('reading file', self.file_path)
            self.data_json = []
            for line in open(self.file_path):
                item = json.loads(line)
                item.pop('edge')
                self.data_json.append(item)

            print('done reading file:', self.file_path)
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_question_len = 80
        self.max_num_sent = 80 
        self.max_num_answers = 8
        self.train_rl = train_rl
        self.train_teacher = train_teacher
        if self.train_rl:
            file_name =   "npy_folder/hotpot_" + str( self.max_seq_len ) + ".memmap"
            self.guide_file = np.memmap(filename = file_name, shape=(len(self.data_json), 2, self.max_seq_len), mode='r', dtype=np.float32)       
            self.train_idxs = pickle.load(open(    "npy_folder/hotpot_" + str(self.max_seq_len) + ".pkl" , 'rb'))
        if self.train_teacher:
            prefix = ''
            self.tot_logits = np.load(prefix+"npy_folder/hotpot_tot_logits.npy")
            self.tot_switch_logits = np.load(prefix+"npy_folder/hotpot_tot_switch_logits.npy")
            self.tot_sent_logits = np.load(prefix+"npy_folder/hotpot_tot_sent_logits.npy")
            self.tot_idxs = pickle.load(open(prefix+"npy_folder/hotpot_tot_idxs.pkl", "rb"))
                        
    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)

    def one_example_to_tensors(self, example, get_idx):
        query_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(example['question'])[: self.max_question_len] + [self.tokenizer.sep_token]
        nodes = []
        title_start = '[unused1]'
        title_end = '[unused2]'
        assert (example['node'][0]['has_supp'])
        assert (example['node'][1]['has_supp'])

        for idx, node in enumerate(example['node']):
            title = node['name'].replace('_', ' ')
            title = self.tokenizer.tokenize(title)            
            title = [title_start] + title + [title_end]
            dL = len(title)
            context = title + node['context']

            spans = []
            for span in node['spans']:
                spans.append ( (span[0] + dL, span[1] + dL  ) )
            
            sent_poses = []
            for sent_pos in node['sent_pos']:  
                sent_poses.append( ( sent_pos[0] + dL, sent_pos[1] + dL - 1) )

            new_node = {'context': context, 'spans':spans, 'sent_pos': sent_poses, 'supp_label': node['supp_label'], 'idx': idx}
            nodes.append(new_node)

    
        max_context = self.max_seq_len - len(query_tokens) - 1
        global DO_GUIDE
        if self.train_teacher:
            global current_ep
            idxs = self.tot_idxs[current_ep][get_idx]
            paras = []
            for r in idxs:
                paras.append(nodes[r])

        elif not DO_GUIDE:

            supports = nodes[:2]
            others = nodes[2:]
            random.shuffle(others)

            current_L = len(supports[0]['context']) + len(supports[1]['context'])
            w = -1
            while current_L < max_context and w+1 < len(others):
                w += 1
                current_L += len(others[w]['context'])

            if w >= 0:
                paras = supports + others[:w] 
                random.shuffle(paras)
                paras.append(others[w])
            else:
                random.shuffle(supports)
                paras = supports

        else:
            paras = []
            idxs = self.train_idxs[get_idx]
            for r in idxs:
                paras.append(nodes[r])


        dL = len(query_tokens)
        start_positions = []
        end_positions = []
        sent_starts = []
        sent_ends = []
        sent_labels = []
        tokens = query_tokens
        for para in paras:
            context = para['context']
            tokens = tokens + context

            for span in para['spans']:
                if span[1] + dL < self.max_seq_len - 1:
                    start_positions.append(span[0] + dL)
                    end_positions.append(span[1] + dL)
            
            for sent_pos,sent_label in zip(para['sent_pos'], para['supp_label']):  
                if sent_pos[1] + dL < self.max_seq_len - 1:
                    sent_starts.append(sent_pos[0] + dL)
                    sent_ends.append(sent_pos[1] + dL)
                    sent_labels.append(sent_label)

            dL += len(context)

        tokens = tokens[: self.max_seq_len -1] + [self.tokenizer.sep_token]

        segment_ids = [0] * (len(query_tokens)) + [1] * (len(tokens)-len(query_tokens))
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

        start_positions = start_positions[:self.max_num_answers]
        end_positions = end_positions[:self.max_num_answers]

        if len(start_positions)==0:
            start_positions = [0]
            end_positions = [0]
        while len(start_positions) < self.max_num_answers:
            start_positions.append(-1)
            end_positions.append(-1)



        sent_starts = sent_starts[:self.max_num_sent]
        sent_ends = sent_ends[:self.max_num_sent]
        sent_labels = sent_labels[:self.max_num_sent]
        
        sent_start_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)
        sent_end_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)

        for idx, (x, y) in enumerate(zip(sent_starts, sent_ends)):
            sent_start_mapping[idx, x] = 1
            sent_end_mapping[idx, y] = 1


        sent_labels.extend( [-1] * (self.max_num_sent-len(sent_labels)) ) 

        answer = example['answer'].lower()

        if answer == "yes":
            switch = 1
        elif answer == "no":
            switch = 2
        else:
            switch = 0

        idxs = [x['idx'] for x in paras]
        item = [torch.tensor(input_ids), 
                                torch.tensor(input_mask),
                                torch.tensor(segment_ids),
                                torch.tensor(start_positions), 
                                torch.tensor(end_positions),
                                torch.tensor(switch),
                                sent_start_mapping,
                                sent_end_mapping,
                                torch.tensor(sent_labels),

                                ]

        if DO_EVAL_GRAD:
            item.append(idxs)
        elif DO_GUIDE:
            item.append( torch.from_numpy(self.guide_file[get_idx]) )
        if self.train_teacher:
            item.append( torch.from_numpy(self.tot_logits[current_ep, get_idx]) )
            item.append( torch.from_numpy(self.tot_switch_logits[current_ep, get_idx]) )
            item.append( torch.from_numpy(self.tot_sent_logits[current_ep, get_idx]) )


        return item


    @staticmethod
    def collate_one_doc_and_lists(batch):
        if not DO_EVAL_GRAD:
            fields = [x for x in zip(*batch)]
            stacked_fields = [torch.stack(field) for field in fields] 
        else:
            num_metadata_fields = 1
            fields = [x for x in zip(*batch)]

            stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
            stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
    

        return stacked_fields





class HotpotDatasetForTest(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, train_rl=False):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print('reading file:', self.file_path)
            self.data_json = []
            for line in open(self.file_path):
                item = json.loads(line)
                self.data_json.append(item)

            print('done reading file:', self.file_path)
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_question_len = 80
        self.max_num_sent = 80 

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)

    def one_example_to_tensors(self, example, idx):
        query_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(example['question'])[: self.max_question_len] + [self.tokenizer.sep_token]
        nodes = []
        title_start = '[unused1]'
        title_end = '[unused2]'
        


        for node in example['node']:
            title = node['name'].replace('_', ' ')
            title = self.tokenizer.tokenize(title)            
            title = [title_start] + title + [title_end]
            dL = len(title)
            context = title + node['context']

            tok_to_orig_index = [-1] * dL  +  [x for x in node['tok_to_orig_index']]
            
            
            sent_poses = []
            for sent_pos in node['sent_pos']:  
                sent_poses.append( ( sent_pos[0] + dL, sent_pos[1] + dL - 1) )

            new_node = {'context': context,  'sent_pos': sent_poses, 'supp_label': node['supp_label'], 'doc_tokens': node['doc_tokens'], 
                            'tok_to_orig_index': tok_to_orig_index }
            nodes.append(new_node)

    
        max_context = self.max_seq_len - len(query_tokens) - 1


        current_L = 0
        w = -1
        paras = []
        while current_L < max_context  and w+1 < len(nodes):
            w += 1
            current_L += len(nodes[w]['context'])
            paras.append(nodes[w])


        dL = len(query_tokens)

        sent_starts = []
        sent_ends = []
        tokens = query_tokens
        tok_to_orig_index = [-1] * len(query_tokens)

        all_doc_tokens = []
        d_docL = 0
        for para in paras:
            context = para['context']
            doc_tokens = para['doc_tokens']
            tokens = tokens + context
            all_doc_tokens = all_doc_tokens + doc_tokens

            for x in para['tok_to_orig_index']:
                if x!=-1:
                    tok_to_orig_index.append(x + d_docL)
                else:
                    tok_to_orig_index.append(-1)
            
            for sent_pos  in para['sent_pos']:  
                if sent_pos[1] + dL < self.max_seq_len - 1:
                    sent_starts.append(sent_pos[0] + dL)
                    sent_ends.append(sent_pos[1] + dL)

            dL += len(context)
            d_docL += len(doc_tokens)

        tokens = tokens[: self.max_seq_len -1] + [self.tokenizer.sep_token]
        tok_to_orig_index = tok_to_orig_index[: self.max_seq_len -1] + [-1]

        segment_ids = [0] * (len(query_tokens)) + [1] * (len(tokens)-len(query_tokens))
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


        sent_starts = sent_starts[:self.max_num_sent]
        sent_ends = sent_ends[:self.max_num_sent]
        
        sent_start_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)
        sent_end_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)

        for idx, (x, y) in enumerate(zip(sent_starts, sent_ends)):
            sent_start_mapping[idx, x] = 1
            sent_end_mapping[idx, y] = 1



        item = [torch.tensor(input_ids), 
                                torch.tensor(input_mask),
                                torch.tensor(segment_ids),
                                sent_start_mapping,
                                sent_end_mapping,
                                example['qid'],
                                tok_to_orig_index,
                                example['answer'].lower(),
                                tokens,
                                all_doc_tokens,
                                ]

        return item


    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 5
        fields = [x for x in zip(*batch)]

        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
    
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
        tb_writer = SummaryWriter("logs/hotpot_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    train_dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
                            
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=HotpotDataset.collate_one_doc_and_lists)



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

    global_step = 1
    epochs_trained = 0


    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_f1 = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):


            model.train()
            batch = tuple(t.to(args.device) for t in batch[:9])

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "switch_labels": batch[5],
                "sent_start_mapping": batch[6],
                "sent_end_mapping": batch[7],
                "sent_labels": batch[8]

            }

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
                if args.local_rank in [-1, 0]  and (args.save_steps > 0 and (global_step % args.save_steps == 0  )):
                    update = True
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('f1', f1)
                        if f1 > best_f1:
                            # exact = results['exact']
                            print ( 'Best f1:', f1)
                            best_f1 = f1
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



def train_rl(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/hotpot_log/" + args.output_dir[args.output_dir.find('/')+1:])


    global DO_GUIDE
    DO_GUIDE = True

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    train_dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length, 
                                    train_rl=args.train_rl)
                            
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=HotpotDataset.collate_one_doc_and_lists)




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
    tr_selector_loss, logging_selector_loss = 0.0, 0.0
    tr_dL, logging_dL = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    # Added here for reproductibility
    set_seed(args)
    best_f1 = 0
    sample_K = 8
    all_copy_rate = 0.0#1.0
    jianshao = all_copy_rate / (t_total*0.1)
    split = int(args.num_train_epochs * args.guide_rate) # For convinence

    for _ in train_iterator:
        
        if _ == split: 
            DO_GUIDE = False 
            
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
                # batch = [torch.cat( [t,t,t], dim=0) for t in batch]
                batch = new_batch

            model.train()


            copy_rate = torch.zeros(bsz*sample_K)   
            if DO_GUIDE:
                copy_rate[:bsz] = 1
                batch = tuple(t.to(args.device) for t in batch[:10])
            else:
                batch = tuple(t.to(args.device) for t in batch[:9])

            copy_rate = copy_rate.to(args.device)          


            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "switch_labels": batch[5],
                "sent_start_mapping": batch[6],
                "sent_end_mapping": batch[7],
                "sent_labels": batch[8],
                "copy_rate": copy_rate, 
            }
            if DO_GUIDE:
                inputs["tokens_prob"] =  batch[9]


            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
 
        
            loss = outputs[0]
            output_selector_loss = outputs[1]
            lens = outputs[6]

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(lens)

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
                rewards[i % bsz].append( reward)
                dLs.append(delta_L)
            mean_dl = sum(dLs)/len(dLs)
            tr_dL += mean_dl

            selector_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                rs = np.array(rs)
                baseline = np.mean(rs)
                # std = np.std(rs)
                for j in range(sample_K):
                    reward = (rs[j] - baseline) #/ std
                    selector_loss += reward * output_selector_loss[j*bsz+i]
            selector_loss = selector_loss / L 
            loss = selector_loss 
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

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
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss


                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL

                if args.local_rank  in [-1, 0]  and (args.save_steps > 0 and global_step % args.save_steps == 0 ):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if f1 > best_f1:
                            print ('Best F1:', f1)
                            best_f1 = f1
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

RawResult = collections.namedtuple("RawResult",
                                  ["unique_id", "start_logits", "end_logits", "switch",
                                  "tok_to_orig_index", "answer", "tokens", "doc_tokens"])

def write_predictions(logger, all_results, max_answer_length, do_lower_case, output_prediction_file):

    """Write final predictions to the json file."""

 
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["feature_index", "start_index", "end_index", "logit"])

    all_predictions = collections.OrderedDict()

    for result in all_results:
        qid = result.unique_id
        tok_to_orig_index = result.tok_to_orig_index
        tokens = result.tokens
        assert(len(tok_to_orig_index)==len(tokens))
        doc_tokens = result.doc_tokens

        prelim_predictions = []

        switch = np.argmax(result.switch)
        if switch > 0:
            if switch==1:
                final_text = "yes"
            else:
                final_text = "no"
        else:

            scores = []
            start_logits = result.start_logits[:len(tokens)]
            end_logits = result.end_logits[:len(tokens)]
            for (i, s) in enumerate(start_logits):
                if tok_to_orig_index[i]==-1:
                    continue
                for (j, e) in enumerate(end_logits[i:i+max_answer_length], i):
                    if tok_to_orig_index[j]==-1:
                        break
                    # scores.append( ( (i, j ), s+e  ) )

                    prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=qid,
                                start_index=i,
                                end_index=j,
                                logit=s+e ))

       
            prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: x.logit,
                    reverse=True)

            pred = prelim_predictions[0]

            tok_tokens = tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = tok_to_orig_index[pred.start_index]
            orig_doc_end = tok_to_orig_index[pred.end_index]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, logger)

        all_predictions[result.unique_id] = (final_text, result.answer)

    f1_scores = [  f1_score(prediction, groundtruth)[0]  for (prediction, groundtruth) in all_predictions.values()]

    return np.mean(f1_scores)


def evaluate(args, model, tokenizer, prefix=""):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)
    dataset = HotpotDatasetForTest(file_path=args.dev_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length)

    eval_sampler = SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,  num_workers=4,
                                    collate_fn=HotpotDatasetForTest.collate_one_doc_and_lists)


    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    raw_results = []
    start_time = timeit.default_timer()
    macs_list = json.load(open('macs_list.json'))
    flops = 0
    bert_flops = 0
    bert_512_flops = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = [t for t in batch]
        L_s = batch[1].sum(dim=1)
        l = int(torch.max(L_s))
        batch[0] = batch[0][:, :l]
        batch[1] = batch[1][:, :l]
        batch[2] = batch[2][:, :l]
        batch[3] = batch[3][:, :, :l]
        batch[4] = batch[4][:, :, :l]

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
                "sent_start_mapping": batch[3].to(args.device),
                "sent_end_mapping": batch[4].to(args.device),

            }

            outputs = model(**inputs)
            batch_start_logits = outputs[0]
            batch_end_logits = outputs[1]
            batch_switch_logits = outputs[2]
            
        qids = batch[5]
        tok_to_orig_indexs = batch[6]
        answers = batch[7]
        tokens = batch[8]
        doc_tokens = batch[9]


        if args.model_type.startswith('auto'):
            hidden_size = 768
            num_labels = 2

            lens = outputs[4]


            times = [1, 5, 6] # (for 1,6)
            

            while (len(lens)+1 <len(times)):
                rt = times[-1]
                times = times[:-1]
                times[-1] += rt
            for i in range(batch[0].size(0)):
                flops += macs_list[l] * times[0]
                for j in range(len(lens)):
                    lj = int(lens[j][i]*l)

                    flops += macs_list[lj] * times[j+1]
                    flops += lj * (hidden_size*32+32*1) 


                flops += hidden_size * num_labels
                bert_flops +=   macs_list[l] * 12 + hidden_size * num_labels
                bert_512_flops += macs_list[512] * 12 + hidden_size * num_labels


        for i in range(batch[0].shape[0]):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            switch = batch_switch_logits[i].detach().cpu().tolist()


            raw_results.append(RawResult(unique_id=qids[i],
                                        start_logits=start_logits,
                                        end_logits=end_logits,
                                        switch=switch,
                                        tok_to_orig_index=tok_to_orig_indexs[i],
                                        answer=answers[i],
                                        tokens=tokens[i],
                                        doc_tokens=doc_tokens[i],
                                        ))
                                        
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    print ('Flops:', 2 * flops / len(dataset) / 1000000.0)
    print ('BERT FLOPS:', 2*bert_flops/len(dataset)/1000000.0)
    print ('Bert_512_flops FLOPS:', 2*bert_512_flops/len(dataset)/1000000.0)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    f1 = write_predictions(logger, raw_results,
                    args.max_answer_length,
                    args.do_lower_case,
                    output_prediction_file)

    return {'f1': f1, 'FLOPs': 2 * flops / len(dataset) / 1000000.0}


def evaluate_grad(args, model, tokenizer, prefix="", evaluate=False):


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
                            
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=HotpotDataset.collate_one_doc_and_lists)


    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    output_dim = [0, 5]
    max_seq_length = args.max_seq_length

    file_name = "npy_folder/hotpot_" + str(max_seq_length) + ".memmap"
    layers_scores = np.memmap(filename = file_name, shape=(len(dataset), len(output_dim), max_seq_length), mode='w+', dtype=np.float32)       
 
    cnt = 0
    tot_idxs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        idxs = batch[9]
        tot_idxs.extend(idxs)

        batch = tuple(t.to(args.device) for t in batch[:9])

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
            "switch_labels": batch[5],
            "sent_start_mapping": batch[6],
            "sent_end_mapping": batch[7],
            "sent_labels": batch[8]

        }


        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[-1][1:] 
        last_val = hidden_states[8]
        grads = torch.autograd.grad(loss, [last_val, ])
        grad_delta = grads[0] 

        for idx_idx, idx in enumerate(output_dim):                
            with torch.no_grad():
                delta = last_val - hidden_states[idx]
                dot = torch.einsum("bli,bli->bl", [grad_delta, delta])
                score = dot.abs()  # (bsz, seq_len)
                score = score.view(-1, max_seq_length).detach()                    

            score = score.cpu().numpy()
            layers_scores[cnt: cnt+score.shape[0], idx_idx] = score
        cnt += score.shape[0]


    file_name = "npy_folder/hotpot_" + str(max_seq_length) + ".pkl"
    pickle.dump(tot_idxs,  open(file_name, "wb"))
    assert(len(tot_idxs)==len(dataset))
    return {}




def evaluate_logits(args, model, tokenizer, prefix="", evaluate=False):


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
                            
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1, 
                                    collate_fn=HotpotDataset.collate_one_doc_and_lists)


    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    max_seq_length = args.max_seq_length
 
    cnt = 0

    tot_train_idxs = []
    tot_logits = []
    tot_switch_logits = []
    tot_sent_logits = []
    tot_idxs = []
    for epoch in range(5):

        all_logits = []
        all_switch_logits = []
        all_sent_logits = []
        all_idxs = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            model.eval()
            with torch.no_grad():
                idxs = batch[9]
                all_idxs.extend(idxs)

                batch = tuple(t.to(args.device) for t in batch[:9])

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                    # "start_positions": batch[3],
                    # "end_positions": batch[4],
                    # "switch_labels": batch[5],
                    "sent_start_mapping": batch[6],
                    "sent_end_mapping": batch[7],
                    # "sent_labels": batch[8]
                }

                outputs = model(**inputs)

                start_logits = outputs[0] 
                end_logits = outputs[1] 
                switch_logits = outputs[2]
                sent_logits = outputs[3] 

                logits = torch.cat([start_logits.unsqueeze(1), end_logits.unsqueeze(1)], dim=1)
                logits = logits.cpu().numpy()
                switch_logits = switch_logits.cpu().numpy()
                sent_logits = sent_logits.cpu().numpy()


                all_logits.append(logits)
                all_switch_logits.append(switch_logits)
                all_sent_logits.append(sent_logits)

        all_logits = np.concatenate(all_logits, axis=0)
        all_switch_logits = np.concatenate(all_switch_logits, axis=0)
        all_sent_logits = np.concatenate(all_sent_logits, axis=0)

        tot_logits.append(np.expand_dims( all_logits,  axis=0))
        tot_switch_logits.append(np.expand_dims( all_switch_logits, axis=0))
        tot_sent_logits.append(np.expand_dims( all_sent_logits, axis=0))
        tot_idxs.append(all_idxs)

    tot_logits = np.concatenate( tot_logits, axis=0)
    tot_switch_logits = np.concatenate( tot_switch_logits, axis=0)
    tot_sent_logits = np.concatenate( tot_sent_logits, axis=0)

    print (tot_logits.shape)

    prefix = ''
    np.save(prefix+"npy_folder/hotpot_tot_logits.npy", tot_logits)
    np.save(prefix+"npy_folder/hotpot_tot_switch_logits.npy", tot_switch_logits)
    np.save(prefix+"npy_folder/hotpot_tot_sent_logits.npy", tot_sent_logits)
    pickle.dump(tot_idxs,  open(prefix+"npy_folder/hotpot_tot_idxs.pkl", "wb"))
    return {}




def train_both(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/hotpot_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    train_dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length, train_teacher=args.train_teacher)
                            
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=2, 
                                    collate_fn=HotpotDataset.collate_one_doc_and_lists)




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
    best_f1 = 0
    sample_K = 8
    all_copy_rate = 0.0#1.0
    jianshao = all_copy_rate / (t_total*0.1)


    for _ in train_iterator:
        global current_ep
        current_ep = _
        
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
                # batch = [torch.cat( [t,t,t], dim=0) for t in batch]
                batch = new_batch

            model.train()

            batch = tuple(t.to(args.device) for t in batch)


            copy_rate = torch.zeros(bsz*sample_K)   


            copy_rate = copy_rate.to(args.device)          

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "switch_labels": batch[5],
                "sent_start_mapping": batch[6],
                "sent_end_mapping": batch[7],
                "sent_labels": batch[8],
                "copy_rate": copy_rate, 
            }

            if args.train_teacher:
                inputs['teacher_logits'] = batch[-3]
                inputs['teacher_switch_logits'] = batch[-2]
                inputs['teacher_sent_logits'] = batch[-1]

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
 
        
            loss = outputs[0]
            output_selector_loss = outputs[1]
            lens = outputs[6]

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(lens)


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
                rewards[i % bsz].append( reward)
                dLs.append(delta_L)
            mean_dl = sum(dLs)/len(dLs)

            selector_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                rs = np.array(rs)
                baseline = np.mean(rs)
                # std = np.std(rs)
                for j in range(sample_K):
                    reward = (rs[j] - baseline) #/ std
                    selector_loss += reward * output_selector_loss[j*bsz+i]
            selector_loss = selector_loss / L 
            loss = selector_loss + qa_loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                mean_dl = mean_dl / args.gradient_accumulation_steps
            tr_dL += mean_dl

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
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar("selector_loss", (tr_selector_loss - logging_selector_loss) / args.logging_steps, global_step)
                    logging_selector_loss = tr_selector_loss


                    tb_writer.add_scalar("dL", (tr_dL - logging_dL) / args.logging_steps, global_step)
                    logging_dL = tr_dL

                if args.local_rank in [-1, 0]  and (args.save_steps > 0 and global_step % args.save_steps == 0 ):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if f1 > best_f1:
                            print ('Best F1:', f1)
                            best_f1 = f1
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


# def drop_train_data(args, tokenizer):
#     args.train_dataset = os.path.join(args.data_dir, args.train_file)
#     train_dataset = HotpotDataset(file_path=args.train_dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_length)
#     features = []
#     for i in range(len(train_dataset)):
#         features.append(train_dataset[i])
        
#     dataset = HotpotDataset.collate_one_doc_and_lists(features)
#     torch.save(dataset, os.path.join(args.data_dir , 'drop_traindataset'))


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
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default="hotpot_train_graph.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default="rerank_dev_tok.json",
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

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
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
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every X updates steps.")
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
                        help="")
    

    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--alpha", default=1.0, type=float, help="")


    parser.add_argument("--drop_data",
                        default=False,
                        action='store_true',
                        help=" ")
    parser.add_argument("--guide_rate",
                        default=0.5,
                        type=float,
                        help="")

    parser.add_argument("--do_eval_logits",
                        default=False,
                        action='store_true',
                        help="")
    

    parser.add_argument("--train_teacher",
                        default=False,
                        action='store_true',
                        help=" ")


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
        create_exp_dir(args.output_dir, scripts_to_save=['run_hotpot.py', 'transformers/src/transformers/modeling_bert.py', 
        'transformers/src/transformers/modeling_rlqabert.py', 'transformers/src/transformers/modeling_autoqabert.py',])

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
        global DO_EVAL_GRAD
        DO_EVAL_GRAD = True
    if args.do_eval_grad:
        DO_EVAL_GRAD = True

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

    # if args.drop_data:
    #     drop_train_data(args, tokenizer)

    # Training
    if args.do_train:
        if args.train_rl:
            global_step, tr_loss = train_rl(args, model, tokenizer)
        elif args.train_both:
            global_step, tr_loss = train_both(args, model, tokenizer)
        else:
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
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    
    
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
        
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True,  config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_grad(args, model, tokenizer, prefix=global_step, evaluate=False)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)


    if args.do_eval_logits and args.local_rank in [-1, 0]:
        
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True,  config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_logits(args, model, tokenizer, prefix=global_step, evaluate=False)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)



    return results


if __name__ == "__main__":
    main()

