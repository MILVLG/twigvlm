from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
import os
from transformers.generation.utils import GenerateOutput
import transformers

from .generator_utils.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategyResult
)
from .generator_utils.speculative_streamer import SpeculativeTextStreamer
from .generator_utils.ssd_generator_with_ttp import SelfSpeculativeGenerationStrategy
from .generator_utils.ar_generator_with_ttp import AutoRegressiveGenerationStrategy
from twigvlm.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)


############################################################################
#           methods to add/replace on original model class                 #
############################################################################
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
    generation_config.enable_pruning = twigvlm_config["enable_pruning"]
    generation_config.exit_layer = int(os.environ.get('twig_K'))
    attention_rank = twigvlm_config["attention_rank"]

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
            # image_sizes=image_sizes
        )
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    # compute the retained visual tokens
    base_T = len(self.model.layers)
    visual_token_num = (image_tags == 1).sum().item()
    attention_rank = math.ceil((base_T*attention_rank-generation_config.exit_layer*visual_token_num)/(generation_config.finalwipe_layer-generation_config.exit_layer))
    generation_config.attention_rank = attention_rank

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

############################################################################
#           methods to add/replace on original model class                 #
############################################################################


def twigvlm_monkey_patch(cls_CausalLM, cls_Model, model_type='llama'):
    """
    Apply monkey patch to the original LlavaXxxxForCausalLM class and LlavaXxxxModel class,
    adapting them to TwigVLM specific methods and attributes.
    """
    # LlavaLlamaModel(LlavaMetaModel, LlamaModel)
    if model_type == 'llama':
        from .twigvlm_patches_for_llama import LlamaModel, patch_modeling
        patch_modeling()
        cls_Model.__bases__ = (LlavaMetaModel, LlamaModel)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    cls_CausalLM.generate = generate
    cls_CausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
    cls_CausalLM.create_logits_processors = create_logits_processors

    return cls_CausalLM