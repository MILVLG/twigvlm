#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaModel, LlamaConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .self_speculation.llama_model_utils_twigvlm import LlamaModel, LlamaConfig, CustomLlamaDecoderLayer
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import time
import colorama

from .self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategyResult
)
from .self_speculation.speculative_streamer import SpeculativeTextStreamer
from .self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from .self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head_early = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head_early_norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.extra_layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx, 4) for layer_idx in range(2,5)]
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    
    def create_logits_processors(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.generation.logits_process.LogitsProcessorList:
        logits_processors: transformers.generation.logits_process.LogitsProcessorList = transformers.generation.logits_process.LogitsProcessorList()
        if generation_config.no_repeat_ngram_size:
            logits_processors.append(transformers.generation.logits_process.NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))

        return logits_processors
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        eos_token_id: Optional[torch.Tensor] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        twigvlm_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', False)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.95)
        image_tags = kwargs.pop("image_tags", None)
        
        generation_config = GenerationConfig()
        generation_config.temperature = temperature
        generation_config.sample = do_sample
        generation_config.top_k = top_k
        generation_config.top_p = top_p
        generation_config.max_steps = max_new_tokens
        generation_config.generation_strategy = twigvlm_config["generation_strategy"]
        generation_config.enable_FastV = twigvlm_config["enable_FastV"]
        generation_config.attention_rank = twigvlm_config["attention_rank"]
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                image_tags,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if generation_config.generation_strategy == "autoregressive":
            generation_strategy: GenerationStrategyResult = AutoRegressiveGenerationStrategy()
        elif generation_config.generation_strategy == "self_speculative":
            generation_strategy: GenerationStrategyResult = SelfSpeculativeGenerationStrategy()
        else:
            raise Exception(
                f"Unsupported generation strategy: {generation_config.generation_strategy}"
            )
        logits_processors = self.create_logits_processors(generation_config=generation_config)
        
        with torch.inference_mode():
            generation_strategy_result = generation_strategy.generate_token_ids(
                model=self,
                input_ids=inputs,
                inputs_embeds=inputs_embeds,
                eos_token_id=eos_token_id,
                generation_config=generation_config,
                logits_processors=logits_processors,
                image_tags=image_tags,
                streamer=streamer,
            )
        if streamer:
            streamer.end()
        return generation_strategy_result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_tags = kwargs.pop("image_tags", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_tags is not None:
            inputs['image_tags'] = image_tags
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
