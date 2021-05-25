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
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    BertForTriviaQuestionAnswering,
    BertConfig,
    BertTokenizer,
    BertForQuestionAnsweringHotpotSeg,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_final_text,
    _compute_softmax,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import json

from hotpot_evaluate_v1 import f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import shutil
import pickle
from tqdm import tqdm
logger = logging.getLogger(__name__)
import collections

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnsweringHotpotSeg, BertTokenizer),
}

 

class HotpotSegQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len, max_segment, data_dir, evaluate=False, train_rl=False):
        print('reading file', file_path)

        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.data = []
        with open(self.file_path, "r", encoding='utf-8') as f:
            self.data_json = []
            for line in open(self.file_path):
                item = json.loads(line)
                item.pop('edge')
                self.data_json.append(item)

        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.max_num_sent = 80 
        
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        self.max_segment = max_segment
        self.zero_item = [0] * self.max_seq_len
        self.one_zero_item = [1] + [0] * (self.max_seq_len-1)

        self.evaluate = evaluate


    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)

    def one_example_to_tensors(self, example, idx):
        nodes = []
        title_start = '[unused2]'
        title_end = '[unused3]'
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

        query_tokens = self.tokenizer.tokenize(example['question'])[: self.max_question_len] 

        max_context = self.max_doc_len - len(query_tokens) - 3 - 2

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
            paras = supports


        dL = 0
        answers = []
        sents = []
        doc_tokens = []
        for para in paras:
            context = para['context']
            doc_tokens = doc_tokens + context

            for span in para['spans']:
                answers.append((span[0] + dL, span[1] + dL))

            for sent_pos, sent_label in zip(para['sent_pos'], para['supp_label']):  
                sents.append((sent_pos[0] + dL, sent_pos[1] + dL, sent_label))

            dL += len(context)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3 - 2
        assert max_tokens_per_doc_slice > 0

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_positions_list = []
        end_positions_list = []
        sent_starts_list = []
        sent_ends_list = []
        sent_labels_list = []

        cnt = 0

        if self.doc_stride < 0:
            self.doc_stride = max_tokens_per_doc_slice
        have_answer = False
        all_doc_tokens = doc_tokens
        for slice_start in range(0, len(all_doc_tokens), self.doc_stride):  
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                + ['[unused0]', '[unused1]'] \
                                                + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 3) 

            assert len(segment_ids) == len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # if self.doc_stride >= 0:  # no need to pad if document is not strided
            #     # Zero-pad up to the sequence length.
            padding_len = self.max_seq_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len

            doc_offset = len(query_tokens) + 4 - slice_start
            start_positions = []
            end_positions = []


            yna_answer = example['answer'].lower()

            if yna_answer == "yes":
                start_positions.append(len(query_tokens)+2)
                end_positions.append(len(query_tokens)+2)
            elif yna_answer == "no":
                start_positions.append(len(query_tokens)+3)
                end_positions.append(len(query_tokens)+3)


            for tok_start_position_in_doc, tok_end_position_in_doc in answers:

                if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                    # this answer is outside the current slice
                    continue
                start_positions.append(tok_start_position_in_doc + doc_offset)
                end_positions.append(tok_end_position_in_doc + doc_offset)

            assert len(start_positions) == len(end_positions)

            
            if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                continue

            # answers from start_positions and end_positions if > self.max_num_answers
            start_positions = start_positions[:self.max_num_answers]
            end_positions = end_positions[:self.max_num_answers]
            
            if len(start_positions)==0:
                start_positions = [-1]   # !! e
                end_positions = [-1]
            else:
                have_answer = True

            # -1 padding up to self.max_num_answers
            padding_len = self.max_num_answers - len(start_positions)
            start_positions.extend([-1] * padding_len)
            end_positions.extend([-1] * padding_len)

            # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
            found_start_positions = set()
            found_end_positions = set()
            for i, (start_position, end_position) in enumerate(zip(start_positions, end_positions)):
                if start_position in found_start_positions:
                    start_positions[i] = -1
                if end_position in found_end_positions:
                    end_positions[i] = -1
                found_start_positions.add(start_position)
                found_end_positions.add(end_position)

            sent_starts = []
            sent_ends = []
            sent_labels = []
            for (sent_start, sent_end, sent_label) in sents:
                if sent_start < slice_start or sent_end > slice_end:
                    continue
                sent_starts.append(sent_start + doc_offset)
                sent_ends.append(sent_end + doc_offset)
                sent_labels.append(sent_label)

            sent_starts = sent_starts[:self.max_num_sent]
            sent_ends = sent_ends[:self.max_num_sent]
            sent_labels = sent_labels[:self.max_num_sent]
            
            sent_start_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)
            sent_end_mapping = torch.zeros((self.max_num_sent, self.max_seq_len), dtype=torch.float32)

            for idx, (x, y) in enumerate(zip(sent_starts, sent_ends)):
                if x >=self.max_seq_len or y>=self.max_seq_len:
                    continue
                sent_start_mapping[idx, x] = 1
                sent_end_mapping[idx, y] = 1


            sent_labels.extend( [-1] * (self.max_num_sent-len(sent_labels)) ) 
            
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            start_positions_list.append(start_positions)
            end_positions_list.append(end_positions)
            sent_starts_list.append(sent_start_mapping.unsqueeze(0))
            sent_ends_list.append(sent_end_mapping.unsqueeze(0))
            sent_labels_list.append(sent_labels)

            cnt += 1
            if cnt >= self.max_segment:
                break

        while cnt < self.max_segment:
            input_ids_list.append(self.zero_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            start_positions_list.append([-1] * self.max_num_answers)
            end_positions_list.append([-1] * self.max_num_answers)

            sent_starts_list.append(torch.zeros((1, self.max_num_sent, self.max_seq_len), dtype=torch.float32))
            sent_ends_list.append(torch.zeros((1, self.max_num_sent, self.max_seq_len), dtype=torch.float32))
            sent_labels_list.append([-1]*  self.max_num_sent)

            cnt += 1

        if not have_answer:
            for i in range(len(start_positions_list)):
                start_positions_list[i][0] = 0
                end_positions_list[i][0] = 0


        item = [torch.tensor(input_ids_list), 
                                torch.tensor(input_mask_list),
                                torch.tensor(segment_ids_list),
                                torch.tensor(start_positions_list), 
                                torch.tensor(end_positions_list),
                                torch.cat(sent_starts_list), 
                                torch.cat(sent_ends_list),
                                torch.tensor(sent_labels_list)
                                ]

        
        return item
        

    @staticmethod
    def collate_one_doc_and_lists(batch):
        
        num_metadata_fields = 0  
        
        fields = [x for x in zip(*batch)]
        if num_metadata_fields > 0:
            stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
            stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
        else:
            stacked_fields = [torch.stack(field) for field in fields]  # don't stack metadata fields

        return stacked_fields





class HotpotSegQADatasetForTest(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len, max_segment, data_dir, evaluate=False, train_rl=False):
        print('reading file', file_path)

        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.data = []
        with open(self.file_path, "r", encoding='utf-8') as f:
            self.data_json = []
            for line in open(self.file_path):
                item = json.loads(line)
                self.data_json.append(item)

        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.max_num_sent = 80 
        
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        self.max_segment = max_segment
        self.zero_item = [0] * self.max_seq_len
        self.one_zero_item = [1] + [0] * (self.max_seq_len-1)

        self.evaluate = evaluate

    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        return self.one_example_to_tensors(entry, idx)

    def one_example_to_tensors(self, example, idx):
        nodes = []
        title_start = '[unused2]'
        title_end = '[unused3]'

        for idx, node in enumerate(example['node']):
            title = node['name'].replace('_', ' ')
            title = self.tokenizer.tokenize(title)            
            title = [title_start] + title + [title_end]
            dL = len(title)
            context = title + node['context']
            tok_to_orig_index = [-1] * dL  +  [x for x in node.get('tok_to_orig_index', [])]
         
            sent_poses = []
            for sent_pos in node['sent_pos']:  
                sent_poses.append( ( sent_pos[0] + dL, sent_pos[1] + dL - 1) )

            new_node = {'context': context, 'sent_pos': sent_poses, 'supp_label': node['supp_label'], 'idx': idx,
                    'doc_tokens': node['doc_tokens'],
                            'tok_to_orig_index': tok_to_orig_index }
            nodes.append(new_node)

        query_tokens = self.tokenizer.tokenize(example['question'])[: self.max_question_len] 

        dL = 0
        d_docL = 0
        all_doc_tokens = []
        ori_doc_tokens = []
        tok_to_orig_index = []
        tokens_list = []

        for para in nodes:
            context = para['context']
            all_doc_tokens = all_doc_tokens + context

            ori_doc_tokens  += para['doc_tokens']

            for x in para['tok_to_orig_index']:
                if x!=-1:
                    tok_to_orig_index.append(x + d_docL)
                else:
                    tok_to_orig_index.append(-1)

            dL += len(context)
            d_docL += len(para['doc_tokens'])
        

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3 - 2
        assert max_tokens_per_doc_slice > 0

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        doc_offset_list = []

        cnt = 0

        if self.doc_stride < 0:
            self.doc_stride = max_tokens_per_doc_slice


        for slice_start in range(0, len(all_doc_tokens), self.doc_stride):  
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                + ['[unused0]', '[unused1]'] \
                                                + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 3) 

            assert len(segment_ids) == len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # if self.doc_stride >= 0:  # no need to pad if document is not strided
            #     # Zero-pad up to the sequence length.
            padding_len = self.max_seq_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len

            doc_offset = len(query_tokens) + 4 - slice_start
            
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            doc_offset_list.append(doc_offset)
            cnt += 1
            tokens_list.append(tokens)
            if cnt >= self.max_segment:
                break
        
        while cnt < self.max_segment:
            input_ids_list.append(self.zero_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            tokens_list.append([])
            cnt += 1


        item = [torch.tensor(input_ids_list), 
                                torch.tensor(input_mask_list),
                                torch.tensor(segment_ids_list),
                                doc_offset_list,
                                example['qid'],
                                tok_to_orig_index,
                                example['answer'].lower(),
                                tokens_list,
                                all_doc_tokens,        
                                ]

        
        return item
        

    @staticmethod
    def collate_one_doc_and_lists_eval(batch):
        num_metadata_fields = 6   

        fields = [x for x in zip(*batch)]
        if num_metadata_fields > 0:
            stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
            stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
        else:
            stacked_fields = [torch.stack(field) for field in fields]  # don't stack metadata fields

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        # assert len(batch) == 1
        # fields_with_batch_size_one = [f[0] for f in stacked_fields]
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
    train_dataset = HotpotSegQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir)
                            

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    num_workers=int(args.output_dir.find('test')==-1), 
                                    sampler=sampler,
                                    collate_fn=HotpotSegQADataset.collate_one_doc_and_lists)
                    


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
    steps_trained_in_current_epoch = 0

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

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch[:8])

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "sent_start_mapping": batch[5],
                "sent_end_mapping": batch[6],
                "sent_labels": batch[7]
            }

            # if args.model_type in ["xlnet", "xlm"]:
            #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            #     if args.version_2_with_negative:
            #         inputs.update({"is_impossible": batch[7]})
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            qa_loss = outputs[0]
            
            loss = qa_loss 
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
                    if args.evaluate_during_training:
                        update = False
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if f1 > best_f1:
                            exact = results['exact']
                            print ('Best Exact:', exact, 'F1:', f1)
                            best_f1 = f1
                            update = True

                    if not args.evaluate_during_training or update:
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


def _get_question_end_index(input_ids, tokenizer):
    eos_token_indices = (input_ids == tokenizer.sep_token_id)
    eos_token_indices = eos_token_indices.nonzero()
    assert eos_token_indices.ndim == 2
    assert eos_token_indices.size(0) == 2 * input_ids.size(0)
    assert eos_token_indices.size(1) == 2
    eos_token_indices = eos_token_indices.view(input_ids.size(0), 2, 2)
    return eos_token_indices[:, 0, 1]

def decode(args, input_ids, start_logits, end_logits, tokenizer, attention_mask, doc_offset_list, qids, tok_to_orig_indexs, std_answers, tokens, doc_tokens):
    # find beginning of document
    question_end_index = _get_question_end_index(input_ids[:, 0, :], tokenizer)

    # bsz x seqlen => bsz x n_best_size
    start_logits_indices = start_logits.topk(k=args.n_best_size, dim=-1).indices
    end_logits_indices = end_logits.topk(k=args.n_best_size, dim=-1).indices

    predictions = collections.OrderedDict()

    # all_answers = []
    # This loop can't be vectorized, so loop over each example in the batch separetly
    for i in range(start_logits_indices.size(0)):  # ins_num
        answers = []

        for j in range(start_logits_indices.size(1)):
            if input_ids[i, j, 1]==0:
                continue
            potential_answers = []
            
            for start_logit_index in start_logits_indices[i,j]:  # n_best_size

                for end_logit_index in end_logits_indices[i,j]:  # n_best_size         
                    if start_logit_index <= question_end_index[i]:
                        continue
                    if end_logit_index <= question_end_index[i]:
                        continue
                    if attention_mask[i,j,start_logit_index]==0:
                        continue
                    if attention_mask[i,j,end_logit_index]==0:
                        continue

                    if input_ids[i,j,end_logit_index]==tokenizer.sep_token_id:
                        continue

                    if start_logit_index > end_logit_index:
                        continue
                    answer_len = end_logit_index - start_logit_index + 1
                    if answer_len > args.max_answer_length:
                        continue
                    potential_answers.append({'start': start_logit_index, 'end': end_logit_index,
                                                'start_logit': start_logits[i,j][start_logit_index].item(),
                                                'end_logit': end_logits[i,j][end_logit_index].item()})
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)
            if len(sorted_answers) == 0:
                answers.append({'text': 'NoAnswerFound', 'score': -1000000, 'output_start':0, 'output_end':0})
            else:
                answer = sorted_answers[0]

                answer_token_ids = input_ids[i, j, answer['start']: answer['end'] + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())

                score = answer['start_logit'] + answer['end_logit']

                if input_ids[i, j, answer['start']]==1:
                    text = 'yes'
                elif input_ids[i, j, answer['start']]==2:
                    text = 'no'
                else:
                    # answer_tokens = tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                    tok_tokens = tokens[i][j][answer['start']: answer['end'] + 1]#tokenizer.convert_tokens_to_string(answer_tokens)
                    d_o = doc_offset_list[i][j]


                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    text = tok_text

                    orig_tokens = doc_tokens[i][answer['start']-d_o:answer['end']-d_o+1]
                    orig_text = " ".join(orig_tokens)
                    text = get_final_text(tok_text, orig_text, do_lower_case=args.do_lower_case)

                answers.append({'text': text, 'score': score, 'answer_start': answer['start'], 'answer_end': answer['end'] })

        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        # all_answers.append(answers[0])
        
        predictions[qids[i]] = (answers[0]['text'], std_answers[i])

    return predictions



def evaluate(args, model, tokenizer, prefix=""):
    # if prefix.find('/')!=-1:
    #     prefix = ''

    # if os.path.exists(os.path.join(args.output_dir, 'prediction_'+prefix+'.json')):
    #     return {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)

    dataset = HotpotSegQADatasetForTest(file_path=args.dev_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir,
                                evaluate=True)

    # Note that DistributedSampler samples randomly    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=HotpotSegQADatasetForTest.collate_one_doc_and_lists_eval)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    flops = 0
    bert_flops = 0
    bert_512_flops = 0
    hidden_size = 768
    num_labels = 2

    predictions = []
    cnt = 0
    all_predictions = collections.OrderedDict()

    macs_list = json.load(open('macs_list.json'))


    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):  
        model.eval()

        with torch.no_grad():
            max_segment = batch[1].shape[1]
            if args.eval_batch_size==1 and max_segment==1:
                l = int(torch.sum(batch[1]))
                batch[0] = batch[0][:,:,:l]
                batch[1] = batch[1][:,:,:l]
                batch[2] = batch[2][:,:,:l]
            else:
                l = batch[1].shape[2]


            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
                
            }

            outputs = model(**inputs)
            start_logits, end_logits = outputs[:2]

            doc_offset_list = batch[3]
            qids = batch[4]
            tok_to_orig_indexs = batch[5]
            answers = batch[6]
            tokens = batch[7]
            doc_tokens = batch[8]


        predictions = decode(args, batch[0].detach(), start_logits.cpu().detach(), end_logits.cpu().detach(), 
            tokenizer, batch[1].detach(), doc_offset_list, qids, tok_to_orig_indexs, answers, tokens, doc_tokens)
        all_predictions.update(predictions)
        if args.eval_batch_size==1:
            flops += (macs_list[l] * 12 + 768*2) * max_segment 
                                        
    f1_scores = [  f1_score(prediction, groundtruth)[0]  for (prediction, groundtruth) in all_predictions.values()]

    f1 = np.mean(f1_scores)

    print ('F1:', f1)
    print ('Flops:', 2*flops / len(dataset) / 1000000.0)

    return {'f1': f1}
    
    

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
        default="naturalqa",
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

  
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
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
        "--num_train_epochs", default=6.0, type=float, help="Total number of training epochs to perform."
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
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=10,
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

    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum length of seq passed to the transformer model")
    parser.add_argument("--max_doc_len", type=int, default=1024,
                        help="Maximum number of wordpieces of the input document")
    parser.add_argument("--max_num_answers", type=int, default=8,
                        help="Maximum number of answer spans per document (64 => 94%)")
    parser.add_argument("--max_question_len", type=int, default=55,
                        help="Maximum length of the question")
    parser.add_argument("--doc_stride", type=int, default=-1,
                        help="Overlap between document chunks.")
    parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                        help="each example should have at least one answer. Default is False")
    parser.add_argument("--test", action='store_true', help="Test only, no training")
    parser.add_argument("--max_segment", type=int, default=2, help="2 for 512 bert")

    # parser.add_argument("--do_eval_grad",
    #                     default=False,
    #                     action='store_true',
    #                     help="Whether to run eval on the dev set.")

    # parser.add_argument("--train_rl",
    #                     default=False,
    #                     action='store_true',
    #                     help=" ")

    # parser.add_argument("--train_both",
    #                     default=False,
    #                     action='store_true',
    #                     help=" ")

    # parser.add_argument("--alpha", default=0.1, type=float, help="")
    # parser.add_argument("--guide_rate", default=0.2, type=float, help="")


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
        create_exp_dir(args.output_dir, scripts_to_save=['run_hotpot_for_512_sharenorm.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_autoqabert.py'])

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
        if args.no_cuda:
            args.n_gpu = 0
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
    # if args.do_eval_grad:
    #     config.output_hidden_states = True

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
        # logger.info("Loading checkpoints saved during training for evaluation")
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)
            print (global_step, result)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))


    return results


if __name__ == "__main__":
    main()
