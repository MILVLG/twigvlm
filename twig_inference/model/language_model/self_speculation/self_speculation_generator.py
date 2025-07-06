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
from ..self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
    GenerationResult
)
from ..self_speculation.speculative_streamer import SpeculativeTextStreamer

from .llama_model_utils_twigvlm import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
    crop_past_key_value_cache,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

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
                    exit_vision_layer=generation_config.exit_vision_layer,
                    eos_token_id=eos_token_id,
                    calls=calls,
                    sample=generation_config.sample,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    logits_processors=logits_processors,
                    prefill_length=prefill_length,
                    enable_FastV=generation_config.enable_FastV,
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
        exit_vision_layer: int,
        reduced_tokens: int = 0,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 0,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        prefill_length: Optional[int] = 0, 
        enable_FastV: Optional[bool] = False,
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
                enable_FastV,
                image_tags,
                _,
                attention_rank
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            past_key_value_shared = draft_result.past_key_value_shared
            draft_logits = draft_result.logits
            # store the keep_indexs for fastv and only through once
            if enable_FastV and draft_result.keep_indexs is not None:
                keep_indexs = draft_result.keep_indexs
                keep_indexs_2 = draft_result.keep_indexs_2
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)

            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            
            # print(draft_next_token)
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
            exit_vision_layer,
            exit_query_cache,
            enable_FastV,
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

        if enable_FastV:
            past_key_values = crop_past_key_values(
                past_key_values=past_key_values, 
                maximum_length=prefill_length+reduced_tokens+len(output_ids) - 1,
                exit_layer=exit_layer,
                exit_vision_layer=exit_vision_layer,
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