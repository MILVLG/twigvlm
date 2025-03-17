# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from dataclasses import dataclass
from typing import List, Optional
import torch
import transformers


@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None


@dataclass
class GenerationResult:
    predicted_tokens: List[int]
    num_tokens_generated: Optional[int] = None
    prefill_time: Optional[float] = None
    acceptance_rate: Optional[float] = None
    prefill_length: Optional[float] = None
    decoding_tokens_per_second: Optional[float] = None
    total_draft_matches: Optional[int] = None
    total_generations: Optional[int] = None
    decoding_time: Optional[float] = None




@dataclass
class GenerationConfig:
    max_steps: int = 512
    exit_layer: int = 2
    num_speculations: int = 5
    generation_strategy: str = "self_speculative"
    sample: bool = False
    temperature: float = 0
    top_k: int = 0
    top_p: float = 0
    no_repeat_ngram_size: int = None
    stop_words: List[str] = None
    enable_FastV: bool = False
    attention_rank: int = 0
    exit_vision_layer: int = 24


class GenerationStrategy:
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,  
    ) -> GenerationResult:
        raise NotImplementedError()


