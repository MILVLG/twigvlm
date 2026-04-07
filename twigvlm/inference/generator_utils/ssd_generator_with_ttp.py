# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple
import colorama
import torch
import time
import math
import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from .generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
    GenerationResult,
    ForwardResult
)
from .speculative_streamer import SpeculativeTextStreamer

from .utils import (
    crop_past_key_values,
    switch_cache,
    PHead_attention_module,
    _prepare_decoder_attention_mask
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values_draft: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    wipe_layer: List[int],
    enable_pruning: bool,
    image_tags: torch.Tensor,
    forward_early_idx: int,
    attention_rank: List[int],
    tree_mask: torch.Tensor,
    top_k: int
) -> ForwardResult:
    device = None
    is_first_forward = False
    keep_indexs = None

    if input_ids is not None:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
    else:
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape
        is_first_forward = True

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values_draft is not None:
        past_key_values_length = past_key_values_draft[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
        
    past_key_values_draft = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values_draft)

    cache_position = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )

    if forward_early_idx > 0:
        cache_position = (cache_position[0]-top_k*(forward_early_idx-1)+forward_early_idx-1).repeat(top_k)

    position_ids = cache_position.unsqueeze(0).view(-1, seq_length)

    if input_ids is not None:
        inputs_embeds = model.model.embed_tokens(input_ids)
    
    attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)

    attention_mask_eager = _prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, tree_mask
    )

    attention_mask = attention_mask_eager

    hidden_states = inputs_embeds
    for layer_id, decoder_layer in enumerate(model.model.layers[:wipe_layer[0]]):
        hidden_states, past_key_values_draft = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            attention_mask_eager=attention_mask_eager,
            position_ids=position_ids,
            past_key_value=past_key_values_draft,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            output_qk=False
            # padding_mask=None,
        )

    # extra_layers
    attention_score = None

    twig_T = len(model.model.twig_layers) 
    for layer_id, decoder_layer in enumerate(model.model.twig_layers):
        if layer_id == twig_T-1 and enable_pruning and is_first_forward:
            #############################################################
            #             Twig-guided Token Pruning (TTP)               #
            #############################################################
            hidden_states, past_key_values_draft, qk = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_eager=attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values_draft,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                output_qk=True
                # padding_mask=None,
            )
            attention_score = PHead_attention_module(model, qk, image_tags)
            top_attention_rank_index = attention_score.topk(attention_rank[0]).indices

            keep_indexs = (image_tags != 1)
            keep_indexs.scatter_(1, top_attention_rank_index, True)
            #############################################################
            #             Twig-guided Token Pruning (TTP)               #
            #############################################################
        else:
            hidden_states, past_key_values_draft = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_eager=attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values_draft,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                output_qk=False
                # padding_mask=None,
            )

    past_key_values_draft = past_key_values_draft.to_legacy_cache()

    hidden_states = model.model.twig_norm(hidden_states)
    logits = model.model.twig_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values_draft, keep_indexs=keep_indexs
    )


# TODO: update forward_remainder(...) to use transformers' new KV cache implementation rather than legacy.
def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    draft_tokens: torch.Tensor,
    past_key_values_target: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    wipe_layer: List[int],
    enable_pruning: Optional[bool],
    keep_indexs: torch.Tensor,
    reduced_tokens: int,
    tree_position_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    image_tags: torch.Tensor,
) -> ForwardResult:
    device = None
    if input_ids is not None:
        device = input_ids.device
        input_ids = input_ids[:,-1:]
        batch_size, seq_length = input_ids.shape
    else:
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape 

    full_seq_length = seq_length + draft_tokens.shape[1]
    full_past_key_values_length = 0
    is_first_forward = False

    if past_key_values_target is None:
        is_first_forward = True
    else:
        full_past_key_values_length = past_key_values_target[-1][0].shape[2] + reduced_tokens

    past_key_values_target = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values_target)
    
    if input_ids is not None:
        input_ids = torch.cat([input_ids, draft_tokens], dim=-1)
        inputs_embeds = model.model.embed_tokens(input_ids)
    else:
        draft_tokens_embeds = model.model.embed_tokens(draft_tokens)
        inputs_embeds = torch.cat([inputs_embeds, draft_tokens_embeds], dim=1)

    cache_position = torch.arange(
        full_past_key_values_length,
        full_past_key_values_length+seq_length,
        dtype=torch.long,
        device=device,
    )  
    cache_position = torch.cat([cache_position, tree_position_ids+full_past_key_values_length+seq_length], dim=0)

    position_ids = cache_position.unsqueeze(0).view(-1, full_seq_length)

    attention_mask = torch.ones(
        (batch_size, full_past_key_values_length+full_seq_length),
        dtype=torch.bool,
        device=device
    )
    
    attention_mask_eager = _prepare_decoder_attention_mask(
        attention_mask, (batch_size, full_seq_length), inputs_embeds, full_past_key_values_length, tree_mask
    )

    attention_mask = attention_mask_eager

    hidden_states = inputs_embeds
    # TODO simplify

    for idx, decoder_layer in enumerate(model.model.layers):
        if enable_pruning:
            if is_first_forward:
                if idx in wipe_layer:
                    hidden_size = hidden_states.shape[2]
                    if wipe_layer.index(idx) != 0:
                        image_tags = image_tags[keep_indexs].unsqueeze(0)
                        keep_indexs = (image_tags != 1)

                    true_tensor = torch.ones(1, hidden_states.shape[1]-keep_indexs.shape[1], dtype=torch.bool, device=hidden_states.device)
                    image_tag_padding = torch.zeros(1, hidden_states.shape[1]-image_tags.shape[1], dtype=torch.int, device=hidden_states.device)
                    keep_indexs = torch.cat((keep_indexs.to(hidden_states.device), true_tensor), dim=1)
                    image_tags = torch.cat((image_tags.to(hidden_states.device), image_tag_padding), dim=1)

                    hidden_states = hidden_states[keep_indexs,:].view(batch_size, -1, hidden_size)
                    position_ids = position_ids.expand(batch_size, -1)[keep_indexs.to(position_ids.device)].view(batch_size, -1)
                    cache_position = cache_position[keep_indexs[0,:].to(cache_position.device)]
                    new_seq_length = hidden_states.shape[1]
                    attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                    attention_mask = attention_mask[:,:,-new_seq_length:, -new_seq_length:]
            else:
                if idx in wipe_layer:
                    new_seq_length = hidden_states.shape[1] + past_key_values_target[idx][0].shape[2]
                    attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:,:,-new_seq_length:, -new_seq_length:]
    
            hidden_states, past_key_values_target = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_eager=attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values_target,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
            )

    past_key_values_target = past_key_values_target.to_legacy_cache()

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_values=past_key_values_target
    )


class SelfSpeculativeGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        eos_token_id: int,
        generation_config: GenerationConfig,
        image_tags: torch.Tensor,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        streamer: Optional[transformers.TextStreamer] = None, 
    ) -> GenerationResult:

        with torch.inference_mode():
            # self-speculative
            calls: int = 0
            prefill_time = 0
            decoding_time = 0
            total_draft_matches = 0
            total_generations = 0
            reduced_tokens = 0
            output_ids: List[int] = []
            past_key_values_draft = None
            past_key_values_target = None
            prefill_length=inputs_embeds.shape[1]
            torch.cuda.synchronize()
            decoding_start = time.time()
            while len(output_ids) < generation_config.max_steps:
                if input_ids is not None:
                    inputs_embeds = None
                (
                    input_ids,
                    output_ids,
                    past_key_values_draft,
                    past_key_values_target,
                    number_of_matches,
                    num_speculations,
                    prefill_length,
                    reduced_tokens,
                ) = self.single_step_speculation(
                    model,
                    image_tags=image_tags,
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    num_speculations=generation_config.num_speculations,
                    past_key_values_draft=past_key_values_draft,
                    past_key_values_target=past_key_values_target,
                    wipe_layer=generation_config.wipe_layer,
                    eos_token_id=eos_token_id,
                    calls=calls,
                    sample=generation_config.sample,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    logits_processors=logits_processors,
                    prefill_length=prefill_length,
                    enable_pruning=generation_config.enable_pruning,
                    reduced_tokens=reduced_tokens,
                    attention_rank=generation_config.attention_rank,
                    streamer=streamer,
                )
                calls += 1
                total_draft_matches += number_of_matches
                total_generations += num_speculations
                eos_found = False
                if calls == 1:
                    # compute decoding speed
                    prefill_time = time.time() - decoding_start
                if eos_token_id in output_ids:
                    # break out of loop when we get an EOS token
                    # remove the EOS token id
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                if eos_found:
                    break
        torch.cuda.synchronize()
        decoding_time = time.time() - decoding_start
        acceptance_rate = total_draft_matches / total_generations
        num_tokens_generated=len(output_ids)
        return GenerationResult(
            predicted_tokens=[output_ids],
            num_tokens_generated=num_tokens_generated,
            prefill_time=prefill_time,
            acceptance_rate=acceptance_rate,
            total_draft_matches=total_draft_matches,
            total_generations=total_generations,
            decoding_time=decoding_time,
            prefill_length=prefill_length+reduced_tokens,
            decoding_tokens_per_second=num_tokens_generated/decoding_time,
        )

    # generate some draft token and one verified token
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        image_tags: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        output_ids: List[int],
        num_speculations: int,
        past_key_values_draft: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        past_key_values_target: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        wipe_layer: List[int],
        reduced_tokens: int = 0,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 0,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        prefill_length: Optional[int] = 0, 
        enable_pruning: Optional[bool] = False,
        attention_rank: Optional[int] = 0,
    ):  
        device = None
        draft_input_ids=None
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        
        if input_ids is not None:
            device = input_ids.device
            prompt_length: int = input_ids.size(1)
            draft_input_ids = input_ids.clone()
        else:
            device = inputs_embeds.device
            prompt_length: int = inputs_embeds.size(1)
        # draft model output token
        draft_output_ids: List[int] = []
        # prepare for attention maps
        keep_indexs = None
        # forward the draft token

        top_k = 10
        total_tokens = 60
        scores_list, parents_list, ss_token= [], [], []
        tree_mask_init = torch.eye(top_k, device=device)[None, None]
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
        parents_list.append(torch.zeros(1, dtype=torch.long, device=device))
        sample_token = torch.tensor([-1], dtype=torch.long, device=device)
        tree_mask = None
        for _ in range(num_speculations):
            draft_result = forward_early(
                model,
                draft_input_ids,
                inputs_embeds,
                past_key_values_draft,
                wipe_layer,
                enable_pruning,
                image_tags,
                _,
                attention_rank,
                tree_mask,
                top_k,
            )
            past_key_values_draft = draft_result.past_key_values
            draft_logits = draft_result.logits
            # store the keep_indexs for fastv and only through once
            if enable_pruning and draft_result.keep_indexs is not None:
                keep_indexs = draft_result.keep_indexs
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)

            ## tree draft
            if _ == 0:
                last_p = draft_logits[:,-1,:].log_softmax(dim=-1)
                top = torch.topk(last_p, top_k, dim=-1)
                topk_index, topk_p = top.indices, top.values
                scores = topk_p[0]
                scores_list.append(scores[None])
                tree_mask = tree_mask_init
                topk_cs_index = torch.arange(top_k, device=device)
            else:
                last_p = draft_logits[0].log_softmax(dim=-1)
                top = torch.topk(last_p, top_k, dim=-1)
                topk_index, topk_p = top.indices, top.values
                cu_scores = topk_p + scores[:, None]
                scores_list.append(cu_scores)
            
                topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
                topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values 

                scores = topk_cs_p
                out_ids = topk_cs_index // top_k
                tree_mask = torch.cat((tree_mask[:, :, out_ids], tree_mask_init), dim=3)

            draft_input_ids = topk_index.view(-1)[topk_cs_index][None] 

            ss_token.append(topk_index)
            bias1 = top_k if _ > 0 else 0
            bias2 = max(0, _ - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
           
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        total_tokens = min(len(scores_list),total_tokens)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents-1, right=False) 

        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]
        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1
        
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(device)

        # delete sample token
        tree_position_ids = tree_position_ids[1:]
        tree_position_ids -= 1
        tree_mask = tree_mask[:,:,1:,1:]
        # retrieve_indices = retrieve_indices[:,1:]
        draft_tokens_copy = draft_tokens.clone()
        draft_tokens = draft_tokens[:,1:]

        # if streamer:
            # if isinstance(streamer, SpeculativeTextStreamer):
                # print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                # streamer.put(draft_output_ids, is_draft=True)
        # verify model

        verify_results = forward_remainder(
            model,
            input_ids,
            inputs_embeds,
            draft_tokens,
            past_key_values_target,
            wipe_layer,
            enable_pruning,
            keep_indexs,
            reduced_tokens,
            tree_position_ids,
            tree_mask,
            image_tags,
        )
        logits = verify_results.logits
        past_key_values_target = verify_results.past_key_values
        # change the prompt_length and prefill_length after fastv
        if keep_indexs is not None:
            logits_length = logits.shape[1]
            prompt_length = logits_length - draft_tokens.shape[1]
            reduced_tokens = prefill_length - logits_length + draft_tokens.shape[1]
            prefill_length = logits_length - draft_tokens.shape[1]

        # only select the logits relevant to what the draft has outputted.
        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        
        logits = logits[:,-draft_tokens.shape[1]:,:]
        logits = logits[0, retrieve_indices]
        retrieve_indices = retrieve_indices[:,1:]
        candidates = draft_tokens_copy[0, retrieve_indices]
        
        posterior_mask = (
            candidates.to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()

        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

        sample_p = logits[best_candidate, accept_length]

        select_indices = (
            retrieve_indices[best_candidate, : accept_length]
        )
        select_indices = select_indices - 1

        input_ids = candidates[None, best_candidate, : accept_length]
        token = torch.argmax(sample_p)

        token = token[None, None]
        input_ids = torch.cat([input_ids, token],dim=1)

        # streamer = True
        # if streamer:
        #     if isinstance(streamer, SpeculativeTextStreamer):
        #         # streamer.delete(len(draft_output_ids[0, :]))
        #         print(colorama.Fore.GREEN, end="")
        #         # print(number_of_matches)
        #         streamer.put(draft_output_ids[0, : number_of_matches])
        #         print(colorama.Style.RESET_ALL, end="")
        #         streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
        #     else:
        #         # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
        #         streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        if enable_pruning:
            past_key_values_target = crop_past_key_values(
                past_key_values=past_key_values_target, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids),
                wipe_layer=wipe_layer,
                attention_rank=attention_rank,
                prefill_length=prefill_length+len(output_ids),
                select_indices=select_indices,
                enable_pruning=True
            )
            past_key_values_draft = crop_past_key_values(
                past_key_values=past_key_values_draft, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids),
                wipe_layer=wipe_layer,
                enable_pruning=False
            )
        else:
            past_key_values_target = crop_past_key_values(
                past_key_values=past_key_values_target, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids),
                wipe_layer=wipe_layer,
                select_indices=select_indices,
                enable_pruning=False
            )
            past_key_values_draft = crop_past_key_values(
                past_key_values=past_key_values_draft, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids),
                wipe_layer=wipe_layer,
                enable_pruning=False
            )

        output_ids.extend(input_ids[0].tolist())
        return (
            input_ids,
            output_ids,
            past_key_values_draft,
            past_key_values_target,
            accept_length,
            num_speculations,
            prefill_length,
            reduced_tokens,
        )