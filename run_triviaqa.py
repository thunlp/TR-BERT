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
    AUTOQABertForTriviaQuestionAnswering,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_final_text
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import json
from triviaqa_utils import evaluation_utils

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import shutil
import pickle
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForTriviaQuestionAnswering, BertTokenizer),
    "autobert": (BertConfig, AUTOQABertForTriviaQuestionAnswering, BertTokenizer),
}

 

class TriviaQADataset(Dataset):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len, max_segment, data_dir, evaluate=False, train_rl=False, train_teacher=False):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print('reading file:', self.file_path)
            self.data_json = json.load(f)
            print('done reading')
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        self.max_segment = max_segment
        self.zero_item = [0] * self.max_seq_len
        self.one_zero_item = [1] + [0] * (self.max_seq_len-1)

        self.evaluate = evaluate
        self.train_rl = train_rl
        if self.train_rl:
            file_name =   "npy_folder/triviaqa_"+str(max_seq_len)+"_"+str(max_segment)+".memmap"
            self.guide_file = np.memmap(filename = file_name, shape=(len(self.data_json), self.max_segment, 2, self.max_seq_len), mode='r', dtype=np.float32)       

        self.train_teacher = train_teacher
        if self.train_teacher:
            self.teacher_logits = np.load('npy_folder/triviaqa_'+str( self.max_seq_len )+'_logits.npy')

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
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        # tensors_list = []

        paragraph_text = example["context"]
        # document_annotation = example['document_annotation']
        
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)


        assert(len(example["qas"])==1)
        qa = example["qas"][0]
        question_text = qa["question"]
        start_position = None
        end_position = None
        # orig_answer_text = None
        answer_spans = []
        for answer in qa["detected_answers"]:
            # orig_answer_text = answer["text"]
            char_spans = answer['char_spans']
            assert(len(char_spans)==1)
            _start, _end = char_spans[0]
            try:
                start_position = char_to_word_offset[_start]
                end_position = char_to_word_offset[_end]
                answer_spans.append({'start': start_position, 'end': end_position})
            except:
                print('Reading example %s failed' % {idx} )
                start_position = 0
                end_position = 0


        # ===== Given an example, convert it into tensors  =============
        query_tokens = self.tokenizer.tokenize(question_text)
        query_tokens = query_tokens[:self.max_question_len]
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        doc_offset_list = []

        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            # hack: the line below should have been `self.tokenizer.tokenize(token')`
            # but roberta tokenizer uses a different subword if the token is the beginning of the string
            # or in the middle. So for all tokens other than the first, simulate that it is not the first
            # token by prepending a period before tokenizing, then dropping the period afterwards
            # sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
            sub_tokens = self.tokenizer.tokenize('. ' + token)[1:] if i > 0 else self.tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if self.max_doc_len > -1:
            all_doc_tokens = all_doc_tokens[:self.max_doc_len]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
        assert max_tokens_per_doc_slice > 0
        # if self.doc_stride < 0:
        #     # negative doc_stride indicates no sliding window, but using first slice
        #     self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_positions_list = []
        end_positions_list = []

        cnt = 0

        if self.doc_stride < 0:
            self.doc_stride = max_tokens_per_doc_slice
        have_answer = False




        for slice_start in range(0, len(all_doc_tokens), self.doc_stride): #max_tokens_per_doc_slice - self.doc_stride
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
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

            doc_offset = len(query_tokens) + 2 - slice_start
            start_positions = []
            end_positions = []
            for answer_span in answer_spans:
                start_position = answer_span['start']
                end_position = answer_span['end']
                tok_start_position_in_doc = orig_to_tok_index[start_position]
                not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
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


            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            start_positions_list.append(start_positions)
            end_positions_list.append(end_positions)
            doc_offset = len(query_tokens) + 2 - slice_start
            doc_offset_list.append(doc_offset)

            cnt += 1
            if cnt >= self.max_segment:
                break

        while cnt < self.max_segment:
            input_ids_list.append(self.zero_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            start_positions_list.append([-1] * self.max_num_answers)
            end_positions_list.append([-1] * self.max_num_answers)

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
                                ]

        if self.train_rl:
            item.append( torch.from_numpy(self.guide_file[idx]) )

        if self.train_teacher:
            item.append( torch.from_numpy(self.teacher_logits[idx]) )

        if self.evaluate:
            item.append(qa['qid'])
            item.append(qa['answers']["Aliases"])

            item.append( doc_offset_list )
            item.append( tok_to_orig_index )
            item.append( doc_tokens )


                                

        return item
        

    @staticmethod
    def collate_one_doc_and_lists(batch):

        fields = [x for x in zip(*batch)]
        if len(fields)>7:
            num_metadata_fields = 5
            stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
            stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        else:
            num_metadata_fields = 0
            stacked_fields = [torch.stack(field) for field in fields]  # don't stack metadata fields


        # num_metadata_fields = 2  # qids and aliases
        # # else:
        # #     num_metadata_fields = 0
        # fields = [x for x in zip(*batch)]
        # stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        # if num_metadata_fields > 0:
        #     stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

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
        tb_writer = SummaryWriter("logs/triviaqa_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    train_dataset = TriviaQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
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
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)
                    


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
            batch = tuple(t.to(args.device) for t in batch[:5])

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
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
                    update = True
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if f1 > best_f1:
                            exact = results['exact']
                            print ('best Exact:', exact, 'F1:', f1)
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

def _get_question_end_index(input_ids, tokenizer):
    eos_token_indices = (input_ids == tokenizer.sep_token_id)
    eos_token_indices = eos_token_indices.nonzero()
    assert eos_token_indices.ndim == 2
    assert eos_token_indices.size(0) == 2 * input_ids.size(0)
    assert eos_token_indices.size(1) == 2
    return eos_token_indices.view(input_ids.size(0), 2, 2)[:, 0, 1]

def decode(args, input_ids, start_logits, end_logits, tokenizer, doc_offset_list, tok_to_orig_index, doc_tokens):
    # find beginning of document
    question_end_index = _get_question_end_index(input_ids[:, 0, :], tokenizer)

    # bsz x seqlen => bsz x n_best_size
    start_logits_indices = start_logits.topk(k=args.n_best_size, dim=-1).indices
    end_logits_indices = end_logits.topk(k=args.n_best_size, dim=-1).indices

    all_answers = []
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
                answers.append({'text': 'NoAnswerFound', 'score': -1000000})
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[i, j, answer['start']: answer['end'] + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                text = tokenizer.convert_tokens_to_string(answer_tokens)

                d_o = doc_offset_list[i][j]
                s = answer['start'].item()-d_o
                e = answer['end'].item()-d_o
                if e < len(tok_to_orig_index[i]):  # [SEP] attention也为1
                    s = tok_to_orig_index[i][s]
                    e = tok_to_orig_index[i][e]
                    orig_tokens = doc_tokens[i][s:e+1]
                    orig_text = " ".join(orig_tokens)
                    text = get_final_text(text, orig_text, do_lower_case=args.do_lower_case)

                score = answer['start_logit'] + answer['end_logit']
                answers.append({'text': text, 'score': score})
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        all_answers.append(answers[0])

    return all_answers


def evaluate(args, model, tokenizer, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)

    dataset = TriviaQADataset(file_path=args.dev_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir,
                                evaluate=True)


    eval_sampler = SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    

    all_f1_scores  = []
    all_em_scores = []


    qid_to_answer_text = {}


    macs_list = json.load(open('macs_list.json'))

    bert_flops = 0
    bert_512_flops = 0
    flops = 0
    hidden_size = 768
    num_labels = 2


    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()
        qids = batch[-5]
        aliases = batch[-4]
        doc_offset_list = batch[-3]
        tok_to_orig_index = batch[-2]
        doc_tokens = batch[-1]   

        with torch.no_grad():
            max_segment = batch[1].shape[1]

            L_s = batch[1].view(-1, batch[1].shape[-1]).sum(dim=1)
            l = int(torch.max(L_s))
            batch[0] = batch[0][:, :, :l]
            batch[1] = batch[1][:, :, :l]
            batch[2] = batch[2][:, :, :l]
            

            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
                
            }

            outputs = model(**inputs)
            start_logits, end_logits = outputs[:2]

        answers = decode(args, batch[0].to(args.device).detach(), start_logits.detach(), end_logits.detach(), tokenizer, \
            doc_offset_list, tok_to_orig_index, doc_tokens)

        f1_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        # TODO: if slow, skip em_scores, and use (f1_score == 1.0) instead
        em_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]

        for i in range(len(qids)):
            qid = qids[i]
            qid_to_answer_text[qid] = answers[i]['text']

        all_em_scores += em_scores
        all_f1_scores += f1_scores
        
        if args.model_type.startswith('auto') and args.per_gpu_eval_batch_size==1:

            all_lens = outputs[2]
            reduction_times = len(all_lens)
            lens = []
            for j in range(reduction_times):
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
        if l>512:
            assert(max_segment==1)
            bert_512_flops += macs_list[512] * 12 * 2 + hidden_size * num_labels * 2  
        else:
            bert_512_flops += macs_list[l] * 12 * 1 + hidden_size * num_labels * 1

    print ('Flops:', 2*flops / len(dataset) / 1000000.0)
    print ('BERT 512 FLOPS:', 2*bert_512_flops/len(dataset)/1000000.0)   
    print ('BERT FLOPS:', 2*bert_flops/len(dataset)/1000000.0)
    sum_em = sum(all_em_scores)
    sum_f1 = sum(all_f1_scores)
    ge = len(all_em_scores)
    em = sum_em / ge
    f1 = sum_f1 / ge

    return {'Exact match': em, 'F1': f1}
    

def evaluate_grad(args, model, tokenizer, prefix=""):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    dataset = TriviaQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir)


    # Note that DistributedSampler samples randomly    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    max_segment = args.max_segment
    max_seq_length = args.max_seq_len

    file_name = "npy_folder/triviaqa_"+str(max_seq_length)+"_"+str(max_segment)+".memmap"
    output_dims = [0, 5]

    layers_scores = np.memmap(filename = file_name, shape=(len(dataset),  max_segment, len(output_dims), max_seq_length), mode='w+', dtype=np.float32)       
    
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()

        inputs = {
            "input_ids": batch[0].to(args.device),
            "attention_mask": batch[1].to(args.device),
            "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
            "start_positions": batch[3].to(args.device),
            "end_positions": batch[4].to(args.device),
        }


        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[-1] 
        hidden_states = hidden_states[1:]

        last_val = hidden_states[8]


        grads = torch.autograd.grad(loss, [last_val,])
        grad_delta = grads[0] 
        for idx_idx, idx in enumerate(output_dims):                
            with torch.no_grad():
                delta = last_val - hidden_states[idx]
                dot = torch.einsum("bli,bli->bl", [grad_delta, delta])
                score = dot.abs()  # (bsz, seq_len)
                score = score.view(-1, max_segment, max_seq_length).detach()

            score = score.cpu().numpy()
            layers_scores[cnt: cnt+score.shape[0], :,idx_idx, :] = score

        cnt += score.shape[0]


    return {}




def evaluate_logits(args, model, tokenizer, prefix=""):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    dataset = TriviaQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir)


    # Note that DistributedSampler samples randomly    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    max_segment = args.max_segment
    max_seq_length = args.max_seq_len
    all_logits = []
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        model.eval()
        bsz, max_segment, _ = batch[0].shape
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
            }


            outputs = model(**inputs)

            start_logits = outputs[0].view(bsz*max_segment, -1)
            end_logits = outputs[1].view(bsz*max_segment, -1)
            logits = torch.cat([start_logits.unsqueeze(1), end_logits.unsqueeze(1)], dim=1)
            logits = logits.cpu().numpy()
            all_logits.append(logits)


    all_logits = np.concatenate(all_logits, axis=0)
    print (all_logits.shape)
    np.save("npy_folder/triviaqa_"+str( max_seq_len )+"logits.npy", all_logits)

    return {}




def train_both(args, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/triviaqa_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)

    dataset = TriviaQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir,
                                train_rl=args.train_rl,
                                train_teacher=args.train_teacher)
              

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size,  
                                    sampler=sampler,  num_workers=int(args.output_dir.find('test')==-1), 
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)
                    


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
    logger.info("  Num examples = %d", len(dataset))
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
    max_segment = args.max_segment
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

            copy_rate = torch.zeros(bsz*sample_K, max_segment)

            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
                "start_positions": batch[3].to(args.device),
                "end_positions": batch[4].to(args.device),
                "copy_rate": copy_rate,

            }
            if args.train_teacher:
                inputs['teacher_logits'] = batch[-1].to(args.device)


            outputs = model(**inputs)
            loss = outputs[0]
            output_selector_loss = outputs[1]
            all_lens = outputs[4]

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(reduction_times):
                l = torch.mean(all_lens[j].view(L, max_segment), dim=-1)
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


                if args.local_rank in [-1, 0] and (args.save_steps > 0 and (global_step % args.save_steps == 0)):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        best_f1 = results['f1']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('Best F1:', best_f1)

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
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/triviaqa_log/" + args.output_dir[args.output_dir.find('/')+1:])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    args.train_dataset = os.path.join(args.data_dir, args.train_file)

    dataset = TriviaQADataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_answers=args.max_num_answers,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment,
                                data_dir=args.data_dir,
                                train_rl=args.train_rl)
              

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size,  
                                    sampler=sampler,  num_workers=int(args.output_dir.find('test')==-1), 
                                    collate_fn=TriviaQADataset.collate_one_doc_and_lists)
                    


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
    logger.info("  Num examples = %d", len(dataset))
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
    max_segment = args.max_segment
    sample_K = 8

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            bsz = batch[0].size(0)
            if sample_K > 1:
                new_batch = []
                for t in batch[:-2]:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat( tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch

            batch = tuple(t.to(args.device) for t in batch)

            copy_rate = torch.zeros(bsz*sample_K, max_segment)  

            if global_step < args.guide_rate * t_total:
                copy_rate[:bsz] = 1

            copy_rate = copy_rate.to(args.device)          

            model.train()

            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2].to(args.device),
                "start_positions": batch[3].to(args.device),
                "end_positions": batch[4].to(args.device),
                "copy_rate": copy_rate,
                "tokens_prob": batch[5].to(args.device),

            }


            outputs = model(**inputs)
            loss = outputs[0]
            output_selector_loss = outputs[1]
            all_lens = outputs[4]

            L = len(loss)
            qa_loss = sum(loss) / L
            reduction_times = len(all_lens)
            lens = []
            for j in range(reduction_times):
                l = torch.mean(all_lens[j].view(L, max_segment), dim=-1)
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


                if args.local_rank in [-1, 0] and (args.save_steps > 0 and (global_step % args.save_steps == 0)):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        best_f1 = results['f1']
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        print ('best_f1:', best_f1)

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
        default="triviaqa",
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
        default="dev.json",
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
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
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

    parser.add_argument("--max_num_answers", type=int, default=64,
                        help="Maximum number of answer spans per document (64 => 94%)")
    parser.add_argument("--max_question_len", type=int, default=55,
                        help="Maximum length of the question")
    parser.add_argument("--doc_stride", type=int, default=-1,
                        help="Overlap between document chunks.")
    parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                        help="each example should have at least one answer. Default is False")

    parser.add_argument("--do_eval_grad",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum length of seq passed to the transformer model, 512 for BERT, 1024 for BERT_1024")
    parser.add_argument("--max_doc_len", type=int, default=-1,
                        help="Maximum number of wordpieces of the input document")
    parser.add_argument("--max_segment", type=int, default=1, help="2 for BERT, 1 for BERT_1024")



    parser.add_argument("--alpha", default=0.1, type=float, help="")
    parser.add_argument("--guide_rate", default=0.2, type=float, help="")


    parser.add_argument("--train_teacher",
                        default=False,
                        action='store_true',
                        help=" ")



    parser.add_argument("--do_eval_logits",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

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
        create_exp_dir(args.output_dir, scripts_to_save=['run_triviaqa.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_autoqabert.py'])

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
            print (global_step, result)
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


