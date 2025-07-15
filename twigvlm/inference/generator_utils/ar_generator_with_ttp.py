# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import torch
import time
import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
    GenerationResult,
    ForwardResult
)
from .utils import (
    decode_next_token,
    _prepare_decoder_attention_mask,
    delete_cache
)

# Our forward_early(...) and forward_remainder(...) functions currently use transformers library's legacy KV cache implementation that is less efficient.
# To ensure an apples to apples comparison, we created this forward function to use in autoregressive decoding to ensure it uses the same KV cache implementation instead.
# FIXME: update forward_early(...) and forward_remainder(...) to use the updated more efficient KV cache implementation.
def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    enable_pruning: bool,
    image_tags: torch.Tensor,
    exit_layer: int, 
    finalwipe_layer: int,
    attention_rank: int
) -> ForwardResult:
    device = None
    is_first_forward = False
    if input_ids is not None:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
    else:
        is_first_forward = True
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

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
    attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)
    if input_ids is not None:
        inputs_embeds = model.model.embed_tokens(input_ids)

    attention_mask_eager = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    if model.model._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    hidden_states = inputs_embeds
    hidden_states_pre = None

    twig_T = len(model.model.twig_layers)
    for layer_id, decoder_layer in enumerate(model.model.layers):
        if layer_id == exit_layer and is_first_forward and enable_pruning:
            hidden_states_pre = hidden_states.clone()
            for idx, ex_decoder_layer in enumerate(model.model.twig_layers):
                if idx == twig_T-1:
                    hidden_states, last_layer_attention, past_key_values = ex_decoder_layer(
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
                else:
                    hidden_states, past_key_values = ex_decoder_layer(
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
            past_key_values = delete_cache(past_key_values, exit_layer)
            past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
            hidden_states = hidden_states_pre
            hidden_size = hidden_states.shape[-1]
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1) # shape: [batch_size, seq_length, seq_length]
            query_tags = (image_tags == -2) # text prompt identifier, shape: [batch_size, seq_length]
            # attn = (last_layer_attention_avg * query_tags[..., None]).sum(dim=1) # shape: [batch_size, seq_length]
            attn = last_layer_attention_avg[:,-1,:]
            image_weight = attn * (image_tags==1) # shape: [batch_size, seq_length]
            top_attention_rank_index = image_weight.topk(attention_rank).indices # shape: [batch_size, ATTENTION_RANK]
            keep_indexs = (image_tags != 1)
            keep_indexs.scatter_(1, top_attention_rank_index, True)
            hidden_states = hidden_states[keep_indexs,:].view(batch_size, -1, hidden_size)
            position_ids = position_ids.expand(batch_size, -1)[keep_indexs].view(batch_size, -1)
            image_tags = image_tags[keep_indexs].unsqueeze(0)
            keep_indexs_2 = (image_tags != 1)
            new_seq_length = hidden_states.shape[1]
            attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]

            # attention_mask = attention_mask[keep_indexs].view(batch_size, -1)
        elif layer_id == finalwipe_layer and is_first_forward and enable_pruning:
            hidden_size = hidden_states.shape[-1]
            hidden_states = hidden_states[keep_indexs_2,:].view(batch_size, -1, hidden_size)
            position_ids = position_ids.expand(batch_size, -1)[keep_indexs_2].view(batch_size, -1)
            new_seq_length = hidden_states.shape[1]
            attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
        
        if not is_first_forward and enable_pruning:
            if layer_id == exit_layer:
                new_seq_length = hidden_states.shape[1] + past_key_values[layer_id][0].shape[2]
                attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
            elif layer_id == finalwipe_layer:
                new_seq_length = hidden_states.shape[1] + past_key_values[layer_id][0].shape[2]
                attention_mask_eager = attention_mask_eager[:,:,-new_seq_length:, -new_seq_length:]
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            attention_mask_eager=attention_mask_eager,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            # padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values
    )

class AutoRegressiveGenerationStrategy(GenerationStrategy):
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
        ensamble: Optional[bool] = False
    ) -> GenerationResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        device = inputs_embeds.device
        prefill_length = inputs_embeds.shape[1]
        past_key_values = None
        output_ids: List[int] = []
        prefill_time = 0
        decoding_time = 0
        start_time = time.time()
        exit_query_cache = None

        for idx in range(generation_config.max_steps):
            model_output = forward(
                model,
                input_ids,
                inputs_embeds,
                past_key_values,
                generation_config.enable_pruning,
                image_tags,
                generation_config.exit_layer,
                generation_config.finalwipe_layer,
                generation_config.attention_rank
            )
            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=generation_config.sample, temperature=generation_config.temperature, top_k=generation_config.top_k, top_p=generation_config.top_p)
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if idx == 0:
                prefill_time = time.time() - start_time
            if next_token == eos_token_id:
                break

            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(device)

        decoding_time = time.time() - start_time
        return GenerationResult(
            predicted_tokens=[output_ids],
            num_tokens_generated=len(output_ids),
            prefill_time=prefill_time,
            prefill_length=prefill_length,
            decoding_time=decoding_time,
            decoding_tokens_per_second=len(output_ids)/decoding_time,
            acceptance_rate=None,
        )