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
    decode_next_token,
    crop_past_key_values,
    crop_past_key_value_cache,
    switch_cache,
    delete_cache,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    past_key_value_shared: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    enable_pruning: bool,
    image_tags: torch.Tensor,
    forward_early_idx: int,
    attention_rank: int
) -> ForwardResult:
    device = None
    is_first_forward = False
    keep_indexs = None
    keep_indexs_2 = None
    if input_ids is not None:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
    else:
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape
        is_first_forward = True

    seq_length_with_past = seq_length
    past_key_values_length = 0
    
    # switch cache
    cache_length = len(past_key_value_shared)

    if cache_length != 0 and forward_early_idx == 0:
        past_key_value_verify = []
        for layer_id in range(exit_layer, exit_layer+cache_length):
            past_key_value_verify.append(past_key_values[layer_id])

        past_key_values = switch_cache(past_key_values, exit_layer, past_key_value_shared)

        past_key_value_shared = past_key_value_verify
    
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
        
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    cache_position = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = cache_position.unsqueeze(0).view(-1, seq_length)
    
    if input_ids is not None:
        inputs_embeds = model.model.embed_tokens(input_ids)
    
    attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)

    attention_mask_eager = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )
    if model.model._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif model.model._use_sdpa:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    hidden_states = inputs_embeds
    for layer_id, decoder_layer in enumerate(model.model.layers[:exit_layer]):
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            attention_mask_eager=attention_mask_eager,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            # padding_mask=None,
        )

    exit_hidden_states = hidden_states
    # extra_layers
    twig_T = len(model.model.twig_layers) 
    for layer_id, decoder_layer in enumerate(model.model.twig_layers):
        if layer_id == twig_T-1 and enable_pruning and is_first_forward:
            #############################################################
            #             Twig-guided Token Pruning (TTP)               #
            #############################################################
            hidden_states, last_layer_attention, past_key_values = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_eager=attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=True,
                use_cache=True,
                cache_position=cache_position,
                # padding_mask=None,
            )
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1) # shape: [batch_size, seq_length, seq_length]
            attn = last_layer_attention_avg[:,-1,:]
            image_tags = image_tags.to(attn.device)
            image_weight = attn * (image_tags == 1) # shape: [batch_size, seq_length]
            top_attention_rank_index = image_weight.topk(attention_rank).indices # shape: [batch_size, ATTENTION_RANK]

            keep_indexs = (image_tags != 1)
            keep_indexs.scatter_(1, top_attention_rank_index, True)
            image_tags = image_tags[keep_indexs].unsqueeze(0)
            keep_indexs_2 = (image_tags != 1)
            #############################################################
            #             Twig-guided Token Pruning (TTP)               #
            #############################################################
        else:
            hidden_states, past_key_values = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_eager=attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                # padding_mask=None,
            )

    # process attention
    
    past_key_values = past_key_values.to_legacy_cache()
    
    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = exit_hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, exit_hidden_states], dim=1)

    hidden_states = model.model.twig_norm(hidden_states)
    logits = model.model.twig_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_values=past_key_values, past_key_value_shared=past_key_value_shared, exit_query_cache=exit_query_cache,keep_indexs=keep_indexs,keep_indexs_2=keep_indexs_2
    )


# TODO: update forward_remainder(...) to use transformers' new KV cache implementation rather than legacy.
def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    past_key_value_shared: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    finalwipe_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    enable_pruning: Optional[bool],
    keep_indexs: torch.Tensor,
    keep_indexs_2: torch.Tensor,
    reduced_tokens: int,
) -> ForwardResult:
    device = None
    if input_ids is not None:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
    else:
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape 
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    is_first_forward = False
    if len(past_key_values) != len(model.model.layers):
        is_first_forward = True
    if past_key_values is not None and past_key_values[0] is not None:
        
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0

        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
    if input_ids is not None:
        inputs_embeds = model.model.embed_tokens(input_ids)


    cache_position = torch.arange(
        full_past_key_values_length + reduced_tokens,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )  

    position_ids = cache_position.unsqueeze(0).view(-1, seq_length)

    attention_mask = torch.ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
        device=device
    )

    early_attention_mask_eager = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, num_tokens_to_generate), inputs_embeds, draft_past_key_values_length
    )
    if model.model._use_flash_attention_2:
        # 2d mask is passed through the layers
        early_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif model.model._use_sdpa:
        # the manual implementation that requires a 4D causal mask in all cases.
        early_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, num_tokens_to_generate),
            inputs_embeds,
            draft_past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        early_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, num_tokens_to_generate), inputs_embeds, draft_past_key_values_length
        )

    full_attention_mask_eager = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, full_past_key_values_length + reduced_tokens
    )

    if model.model._use_flash_attention_2:
        # 2d mask is passed through the layers
        full_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif model.model._use_sdpa:
        # the manual implementation that requires a 4D causal mask in all cases.
        full_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            full_past_key_values_length + reduced_tokens,
        )
    else:
        # 4d mask is passed through the layers
        full_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, full_past_key_values_length + reduced_tokens
        )

    next_decoder_cache = []
    hidden_states = inputs_embeds

    # TODO simplify

    full_hidden_states: Optional[torch.FloatTensor] = None
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer
        past_key_value = (
            past_key_values[idx]
            if (past_key_values is not None and idx < len(past_key_values))
            else None
        )
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            early_position_ids = position_ids[:, -num_tokens_to_generate:]
            early_cache_position = cache_position[-num_tokens_to_generate:]
            hidden_states, past_key_values = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                attention_mask_eager=early_attention_mask_eager,
                position_ids=early_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=early_cache_position,
                # padding_mask=None,
            )
            # skip it when draft tokens is none
            # update draft model's past_key_values

            if idx == exit_layer - 1 and (input_ids is None or input_ids.shape[1] != 1):
                for layer_id, decoder_layer in enumerate(model.model.twig_layers):
                    early_position_ids = position_ids[:, -num_tokens_to_generate:]
                    early_cache_position = cache_position[-num_tokens_to_generate:]
                    if layer_id == 0:
                        early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
                    else:
                        early_hidden_states = _[:, -num_tokens_to_generate:]
                    _, past_key_values = decoder_layer(
                        early_hidden_states,
                        attention_mask=early_attention_mask,
                        attention_mask_eager=early_attention_mask_eager,
                        position_ids=early_position_ids,
                        past_key_value=past_key_values,
                        output_attentions=False,
                        use_cache=True,
                        cache_position=early_cache_position,
                        # padding_mask=None,
                    )
        else:   
            if full_hidden_states is None and exit_query_cache is not None:
                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
            
            # switch cache or delete cache
            if idx == exit_layer:
                past_key_value_draft = []
                if is_first_forward == False: # shared = verify
                    if full_hidden_states.shape[1] != 1:
                        cache_length = len(past_key_value_shared)
                        for layer_id in range(exit_layer, exit_layer+cache_length):
                            past_key_value_draft.append(past_key_values[layer_id])
                        past_key_values = switch_cache(past_key_values, exit_layer, past_key_value_shared)
                    new_seq_length = full_hidden_states.shape[1] + past_key_values[idx][0].shape[2]
                    full_attention_mask_eager = full_attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                    if full_attention_mask is not None:
                        full_attention_mask = full_attention_mask[:,:,-new_seq_length:, -new_seq_length:]
                else: # delete cache
                    # shared = draft
                    cache_length = len(past_key_values) - exit_layer
                    for layer_id in range(exit_layer, exit_layer+cache_length):
                        past_key_value_draft.append(past_key_values[layer_id])
                    past_key_values = delete_cache(past_key_values, exit_layer)

                past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
            if idx == finalwipe_layer:
                if is_first_forward is False:
                    new_seq_length = full_hidden_states.shape[1] + past_key_values[idx][0].shape[2]
                    full_attention_mask_eager = full_attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                    if full_attention_mask is not None:
                        full_attention_mask = full_attention_mask[:,:,-new_seq_length:, -new_seq_length:]
            if enable_pruning and keep_indexs is not None and is_first_forward:
                if idx == exit_layer:
                    hidden_size = full_hidden_states.shape[2]
                    with torch.no_grad():
                        true_tensor = torch.ones(1, full_hidden_states.shape[1]-keep_indexs.shape[1], dtype=torch.bool, device=full_hidden_states.device)
                        keep_indexs = torch.cat((keep_indexs.to(full_hidden_states.device), true_tensor), dim=1)
                    full_hidden_states = full_hidden_states[keep_indexs,:].view(batch_size, -1, hidden_size)
                    position_ids = position_ids.expand(batch_size, -1)[keep_indexs.to(position_ids.device)].view(batch_size, -1)
                    cache_position = cache_position[keep_indexs[0,:].to(cache_position.device)]
                    new_seq_length = full_hidden_states.shape[1]
                    full_attention_mask_eager = full_attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                elif idx == finalwipe_layer:
                    hidden_size = full_hidden_states.shape[2]
                    with torch.no_grad():
                        true_tensor = torch.ones(1, full_hidden_states.shape[1]-keep_indexs_2.shape[1], dtype=torch.bool, device=full_hidden_states.device)
                        keep_indexs_2 = torch.cat((keep_indexs_2.to(full_hidden_states.device), true_tensor), dim=1)
                    full_hidden_states = full_hidden_states[keep_indexs_2,:].view(batch_size, -1, hidden_size)
                    position_ids = position_ids.expand(batch_size, -1)[keep_indexs_2.to(position_ids.device)].view(batch_size, -1)
                    cache_position = cache_position[keep_indexs_2[0,:].to(cache_position.device)]
                    new_seq_length = full_hidden_states.shape[1]
                    full_attention_mask_eager = full_attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
                    
            hidden_states, past_key_values = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                attention_mask_eager=full_attention_mask_eager,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                # padding_mask=None,
            )

    past_key_values = past_key_values.to_legacy_cache()

    # shared = draft
    past_key_value_shared = past_key_value_draft

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_value_shared=past_key_value_shared, past_key_values=past_key_values, exit_query_cache=exit_query_cache
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
            past_key_values = None
            past_key_value_shared = []
            prefill_length=inputs_embeds.shape[1]
            decoding_start = time.time()
            while len(output_ids) < generation_config.max_steps:
                if input_ids is not None:
                    inputs_embeds = None
                (
                    input_ids,
                    output_ids,
                    past_key_values,
                    past_key_value_shared,
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
                    num_speculations=min(
                        generation_config.num_speculations,
                        generation_config.max_steps - len(output_ids) - 1,
                    ),
                    past_key_values=past_key_values,
                    past_key_value_shared=past_key_value_shared,
                    exit_layer=generation_config.exit_layer,
                    finalwipe_layer=generation_config.finalwipe_layer,
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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        past_key_value_shared: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        finalwipe_layer: int,
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
        # store the hidden_states
        exit_query_cache = None
        # prepare for attention maps
        keep_indexs = None
        keep_indexs_2 = None
        # forward the draft token

        for _ in range(num_speculations):
            draft_result = forward_early(
                model,
                draft_input_ids,
                inputs_embeds,
                past_key_values,
                past_key_value_shared,
                exit_layer,
                exit_query_cache,
                enable_pruning,
                image_tags,
                _,
                attention_rank
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            past_key_value_shared = draft_result.past_key_value_shared
            draft_logits = draft_result.logits
            # store the keep_indexs for fastv and only through once
            if enable_pruning and draft_result.keep_indexs is not None:
                keep_indexs = draft_result.keep_indexs
                keep_indexs_2 = draft_result.keep_indexs_2
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)

            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(device)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break
            if draft_logits[:,-1,:].softmax(dim=-1)[:,draft_next_token] < 0.6:
                break

        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(device)
        prefill_inputs_embeds=None
        prefill_token_ids=None

        if input_ids is not None:
            # the next few times
            prefill_token_ids = torch.cat(
                [input_ids, draft_output_ids],
                dim=-1,
            ).int()
        else:
            # the first forward
            draft_output_embeds = model.model.embed_tokens(draft_output_ids)
            prefill_inputs_embeds = torch.cat([inputs_embeds, draft_output_embeds], dim=1)
    
        # if streamer:
            # if isinstance(streamer, SpeculativeTextStreamer):
                # print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                # streamer.put(draft_output_ids, is_draft=True)
        # verify model
        verify_results = forward_remainder(
            model,
            prefill_token_ids,
            prefill_inputs_embeds,
            past_key_values,
            past_key_value_shared,
            exit_layer,
            finalwipe_layer,
            exit_query_cache,
            enable_pruning,
            keep_indexs,
            keep_indexs_2,
            reduced_tokens,
        )
        logits = verify_results.logits
        past_key_value_shared = verify_results.past_key_value_shared
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # change the prompt_length and prefill_length after fastv
        if keep_indexs is not None:
            logits_length = logits.shape[1]
            prompt_length = logits_length - draft_output_ids.shape[1]
            reduced_tokens = prefill_length - logits_length + draft_output_ids.shape[1]
            prefill_length = logits_length - draft_output_ids.shape[1]
        # only select the logits relevant to what the draft has outputted.
            
        verification_logits = logits[:, prompt_length - 1:, :]

        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(device)
        # print(verified_tokens)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]
        
        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]

        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        # streamer = True
        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                # streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                # print(number_of_matches)
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence

        if enable_pruning:
            past_key_values = crop_past_key_values(
                past_key_values=past_key_values, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids) - 1,
                exit_layer=exit_layer,
                finalwipe_layer=finalwipe_layer,
                length_after_fastv=prefill_length+attention_rank+len(output_ids) - 1,
                length_after_fastv_2=prefill_length+len(output_ids) - 1
            )
            past_key_value_shared = crop_past_key_value_cache( # draft
                past_key_value=past_key_value_shared,
                maximum_length=prefill_length+reduced_tokens+len(output_ids) - 1,
                length_after_fastv=prefill_length+reduced_tokens+len(output_ids) - 1
            )

        else:
            past_key_values = crop_past_key_values(
                past_key_values=past_key_values, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids) - 1,
                exit_layer=exit_layer
            )
            past_key_value_shared = crop_past_key_value_cache( # draft
                past_key_value=past_key_value_shared,
                maximum_length=prefill_length+reduced_tokens+len(output_ids) - 1,
            )

        return (
            input_ids,
            output_ids,
            past_key_values,
            past_key_value_shared,
            number_of_matches,
            draft_output_ids.numel(),
            prefill_length,
            reduced_tokens,
        )