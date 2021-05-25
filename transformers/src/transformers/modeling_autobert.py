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
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .modeling_bert import BertEmbeddings, load_tf_weights_in_bert, BertLayerNorm, BertPreTrainedModel, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, mish, ACT2FN
from .modeling_bert import BertEncoder as OriBertEncoder
from .modeling_bert import BertModel as OriBertModel



logger = logging.getLogger(__name__)




class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        ori_attention_probs = attention_probs
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, ori_attention_probs)
        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        # if encoder_hidden_states is None:
        #     outputs = (context_layer, ori_attention_probs)
        # else:
        #     outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs



class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.linear_size = 32
        # self.minimum_charges = [48, 24, -1]
        self.linear_1 = nn.ModuleList([nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size)
                                       ])

        self.linear_2 = nn.ModuleList([nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1)
                                      ])

    def get_device_of(self, tensor):
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    def get_range_vector(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_ones(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.FloatTensor(size, device=device).fill_(1)
        else:
            return torch.ones(size)

    def get_zeros(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).zero_()
        else:
            return torch.zeros(size, dtype=torch.long)

    def flatten_and_batch_shift_indices(self, indices, sequence_length):
        offsets = self.get_range_vector(indices.size(0), self.get_device_of(indices)) * sequence_length
        for _ in range(len(indices.size()) - 1):
            offsets = offsets.unsqueeze(1)

        # Shape: (batch_size, d_1, ..., d_n)
        offset_indices = indices + offsets

        # Shape: (batch_size * d_1 * ... * d_n)
        offset_indices = offset_indices.view(-1)
        return offset_indices

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        seq_mask=None,
        tokens_prob=None,
        copy_rate=None,
        prob_delta=None,
        is_origin=None
    ):

        all_hidden_states = ()
        all_attentions = ()

        bsz, ori_num_items, dim = hidden_states.size()

        if is_origin:
            for i, layer_module in enumerate(self.layer):
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
                )
                hidden_states = layer_outputs[0]
            return (hidden_states, )



        device = self.get_device_of(hidden_states)

        tot_zoom = None

        tot_select_loss = 0
        Ls = []

        w = 0

        mini = 1
        mid = len(self.layer)//2
        output_probs = []
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            num_items = hidden_states.size(1)
            
            if (i in [1, mid]) and (num_items > mini):

                h1 = gelu(self.linear_1[w](hidden_states))
                h2 = self.linear_2[w](h1)
                prob = torch.sigmoid(h2).squeeze(-1) * seq_mask

                if copy_rate is not None:
                    if copy_rate.sum()>0:
                        if prob_delta is not None:
                            prob_plus = torch.min(prob+prob_delta, torch.ones(prob.shape).to(prob))
                        else:
                            prob_plus = prob

                        with torch.no_grad():
                            if tot_zoom is None:
                                guide = tokens_prob[:,w,:]   # (bsz, 8, seq_len)
                            else:
                                guide = torch.matmul(tot_zoom, tokens_prob[:, w, :].unsqueeze(-1)).squeeze(-1)

                            guide = guide * seq_mask
                            prob_L = torch.sum(prob_plus, dim=-1)
                            prob_L = (prob_L+0.5).int()

                            num_items_to_keep = torch.max(prob_L)

                            top_values, top_indices = guide.topk(num_items_to_keep, 1)                    
                            flatten_indices = self.flatten_and_batch_shift_indices(top_indices, num_items)

                            if device > -1:
                                rt = torch.cuda.FloatTensor(bsz, num_items_to_keep, device=device).zero_()
                                t_p = torch.cuda.FloatTensor(bsz * num_items, device=device).zero_()
                            else:
                                rt = torch.zeros(bsz, num_items_to_keep)
                                t_p = torch.zeros(bsz * num_items)

                            for k in range(bsz):
                                rt[k, :prob_L[k]] = 1

                            t_p[flatten_indices] = rt.view(-1)#1
                            t_p = t_p.view(bsz, num_items)

                        copy_mask = copy_rate.unsqueeze(-1)
                        sample_prob = prob_plus * (1-copy_mask) + t_p * copy_mask
                        
                    else:
                        sample_prob = prob

                    sample_prob = sample_prob.clone()
                    sample_prob[:, 0] = 1

                    m = torch.distributions.bernoulli.Bernoulli(sample_prob)               
                    selected_token = m.sample() * seq_mask

                    select_loss = - ( selected_token * torch.log(prob+1e-6) + (1 - selected_token) * torch.log(1 - prob +1e-6 ) ) * seq_mask
                    select_loss = torch.sum(select_loss, dim=-1) /  torch.sum(seq_mask, dim=-1)         
                    tot_select_loss += select_loss

                else:
                    sample_prob = prob.clone()
                    sample_prob[:, 0] = 3
                    selected_token = (sample_prob >= 0.5).to(seq_mask) * seq_mask

                l = torch.sum(selected_token, dim=-1) 
 
                Ls.append(l / ori_num_items)

                num_items_to_keep = int((torch.max(l)).item())

                if copy_rate is not None:

                    with torch.no_grad():
                        if copy_rate is None:
                            top_values, top_indices = sample_prob.topk(num_items_to_keep, 1)    
                        else:
                            selected_token_2 = selected_token.clone()
                            selected_token_2[:, 0] = 2
                            top_values, top_indices = selected_token_2.topk(num_items_to_keep, 1)    


                                            
                        if device > -1:
                            zoomMatrix = torch.cuda.FloatTensor(bsz * num_items_to_keep, num_items, device=device).zero_()
                        else:
                            zoomMatrix = torch.zeros(bsz * num_items_to_keep, num_items)

                        idx = self.get_range_vector(bsz*num_items_to_keep, device)
                        zoomMatrix[idx, top_indices.view(-1)] = 1.
                        zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)

                        if copy_rate is not None and copy_rate[0]>0:
                            if tot_zoom is None:
                                tot_zoom = zoomMatrix
                            else:
                                tot_zoom = torch.matmul(zoomMatrix, tot_zoom)

                    seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))
                    seq_mask = seq_mask.squeeze(-1) 
                    hidden_states = torch.matmul(zoomMatrix, hidden_states)        

                else:
                    top_values, top_indices = sample_prob.topk(num_items_to_keep, 1)    

                    seq_mask = seq_mask[torch.arange(bsz).unsqueeze(-1), top_indices]
                    hidden_states = hidden_states[torch.arange(bsz).unsqueeze(-1), top_indices]


                if copy_rate is not None:
                    top_values = (top_values>0).to(seq_mask)
                    seq_mask = seq_mask * top_values

                # if copy_rate is None:
                #     output_probs.append(top_indices)



                attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
                w += 1




            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
            # att_prob = layer_outputs[1] 
            

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)




        # for i, layer_module in enumerate(self.layer):
        #     if self.output_hidden_states:
        #         all_hidden_states = all_hidden_states + (hidden_states,)

        #     num_items = hidden_states.size(1)

        #     if (i in [1, mid]) and (num_items > mini):
        #         h1 = gelu(self.linear_1[w](hidden_states))
        #         h2 = self.linear_2[w](h1)
        #         prob = torch.sigmoid(h2).squeeze(-1) * seq_mask

     
        #         selected_token = (prob >= 0.5).to(seq_mask) * seq_mask
        #         l = torch.sum(selected_token, dim=-1) 


        #         num_items_to_keep = int((torch.max(l)).item())

        #         top_values, top_indices = prob.topk(num_items_to_keep, 1)    

        #         seq_mask = seq_mask[torch.arange(bsz).unsqueeze(-1), top_indices]
        #         hidden_states = hidden_states[torch.arange(bsz).unsqueeze(-1), top_indices]


        #         attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        #         attention_mask = (1.0 - attention_mask) * -10000.0
        #         w += 1




        #     layer_outputs = layer_module(
        #         hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
        #     )
        #     hidden_states = layer_outputs[0]

            

        #     if self.output_attentions:
        #         all_attentions = all_attentions + (layer_outputs[1],)





        # for i, layer_module in enumerate(self.layer):
        #     if self.output_hidden_states:
        #         all_hidden_states = all_hidden_states + (hidden_states,)

        #     num_items = hidden_states.size(1)
            

        #     if i==4:
                
        #         # print (tokens_prob.shape)
        #         # print (hidden_states.shape)

        #         # print (seq_mask.shape)
        #         prob = tokens_prob[:,i-1,:]   # (bsz, 8, seq_len)
        #         # prob = torch.rand_like(prob)


        #         # max_seq_L = torch.sum(seq_mask, dim=-1)
        #         # prob = torch.sum(torch.sum(att_prob, dim=1), dim=1)
        #         prob = prob / torch.max(prob)
        #         # prob = torch.rand_like(prob)
 
        #         sample_prob = prob.clone()
        #         sample_prob[:, 0] += 3
        #         keep_rate = 0.1
        #         print (keep_rate)
        #         num_items_to_keep = int(keep_rate*num_items+0.5) 

        #         top_values, top_indices = sample_prob.topk(num_items_to_keep, 1)    

        #         seq_mask = seq_mask[torch.arange(bsz).unsqueeze(-1), top_indices]
        #         hidden_states = hidden_states[torch.arange(bsz).unsqueeze(-1), top_indices]



        #         attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        #         attention_mask = (1.0 - attention_mask) * -10000.0
        #         w += 1




        #     layer_outputs = layer_module(
        #         hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
        #     )
        #     hidden_states = layer_outputs[0]
        #     att_prob = layer_outputs[1] 
            

        #     if self.output_attentions:
        #         all_attentions = all_attentions + (layer_outputs[1],)






        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        outputs = (hidden_states, tot_select_loss, Ls)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



BERT_START_DOCSTRING = r"""
"""

BERT_INPUTS_DOCSTRING = r"""
"""


class AUTOBertModel(BertPreTrainedModel):
 
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder =  BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        score=None,
        tokens_prob=None,
        copy_rate=None,
        prob_delta=None,
        is_origin=None,
        # is_training=None,        
    ):
       

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        seq_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            seq_mask=seq_mask,
            tokens_prob=tokens_prob,
            copy_rate=copy_rate,
            prob_delta=prob_delta,
            is_origin=is_origin,
            # is_training=is_training
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)




@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class AUTOBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = AUTOBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        tokens_prob=None,
        copy_rate=None,
        prob_delta=None,
    ):
        bsz =  input_ids.shape[0]
        num_choices = input_ids.shape[1]


        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        if copy_rate is not None:
            copy_rate = copy_rate.view(-1)

        if tokens_prob is not None:
            sizes = list(tokens_prob.size())
            tokens_prob = tokens_prob.view( [-1] + sizes[2:] )  

            
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            tokens_prob=tokens_prob,
            copy_rate=copy_rate,
            prob_delta=prob_delta
        )

        pooled_output = outputs[1]

        if copy_rate is not None:
            selector_loss = outputs[2]
            selector_loss = selector_loss.view(bsz, -1) # (bsz, 4)
            selector_loss = torch.mean(selector_loss, dim=-1)
            # selector_loss = selector_loss[torch.arange(bsz), labels]  # bsz, 

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[3:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss, selector_loss) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class AUTOBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        if config.temperature:
            self.temperature = config.temperature
        else:
            self.temperature = 1
        self.bert = AUTOBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        tokens_prob=None,
        copy_rate=None,
        teacher_logits=None,
        is_origin=None,
        # is_training=False,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            tokens_prob=tokens_prob,
            copy_rate=copy_rate,
            is_origin=is_origin,
            # is_training=is_training,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if copy_rate is not None:
            selector_loss = outputs[2]

        outputs = (logits,) + outputs[3:]  # add hidden states and attention if they are here


        if teacher_logits is not None:
            # soft_criterion = nn.KLDivLoss(reduction='none')
            teacher_probs = nn.functional.softmax(teacher_logits/self.temperature)
            student_log_probs = nn.functional.log_softmax(logits.view(-1, self.num_labels)/self.temperature)
            loss = -student_log_probs * teacher_probs

            # loss = soft_criterion(student_probs, teacher_probs) 
            # loss = soft_criterion(student_log_probs, teacher_probs) 
            # loss = student_probs * torch.log(student_probs/teacher_probs)
            loss = torch.sum(loss, dim=-1)
            outputs = (loss, selector_loss) + outputs

        elif labels is not None:
            # if self.num_labels == 1:
            #     #  We are doing regression
            #     loss_fct = MSELoss()
            #     loss = loss_fct(logits.view(-1), labels.view(-1))
            # else:
            if is_origin:
                return (logits.view(-1, self.num_labels), )

            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, selector_loss) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
