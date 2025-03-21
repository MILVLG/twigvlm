# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional

import torch
import time
import transformers
from ..self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
    GenerationResult
)
from .llama_model_utils_twigvlm import (
    decode_next_token,
    forward,
    forward_early,
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
                generation_config.enable_FastV,
                image_tags,
                generation_config.exit_layer,
                generation_config.exit_vision_layer,
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