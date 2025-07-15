from typing import List, Optional, Tuple, Union

import torch
import transformers
from .generator_base import ForwardResult

# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        raise NotImplementedError("Sampling is not implemented yet.")
        if not token_idx:
            logits.squeeze_(dim=0)

        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token.transpose_(1, 0)
        return next_token, probabilities

    
def switch_cache(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    switch_layer: Optional[int],
    switch_past_key_value: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    cache_length = len(switch_past_key_value)
    for idx in range(len(past_key_values)):
        if idx >= switch_layer and idx < switch_layer + cache_length:
            new_past.append(switch_past_key_value[idx-switch_layer])
        else:
            new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
    return tuple(new_past)

def delete_cache(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    delete_layer: Optional[int],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(past_key_values)):
        if idx < delete_layer:
            new_past.append((past_key_values[idx][0], past_key_values[idx][1]))
    return tuple(new_past)

def crop_past_key_value_cache(
    past_key_value: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
    length_after_fastv: Optional[int] = None, 
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(len(past_key_value)):
        if length_after_fastv is None:
            new_cache.append(
                (
                    past_key_value[_][0][:,:,:maximum_length,:], 
                    past_key_value[_][1][:,:,:maximum_length,:]
                )
            )
        else:
            new_cache.append(
                (
                    past_key_value[_][0][:,:,:length_after_fastv,:], 
                    past_key_value[_][1][:,:,:length_after_fastv,:]
                )
            )
    past_key_value = tuple(new_cache)
    return past_key_value

def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
    exit_layer: Optional[int],
    finalwipe_layer: Optional[int] = -1, 
    length_after_fastv: Optional[int] = None,  
    length_after_fastv_2: Optional[int] = None,  
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if length_after_fastv is None: # no fastv
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                )
            )
    else: # fastv
        for idx in range(len(past_key_values)):
            if idx < exit_layer:
                new_past.append(
                    (
                        past_key_values[idx][0][:, :, :maximum_length, :],
                        past_key_values[idx][1][:, :, :maximum_length, :],
                    )
                )
            elif idx < finalwipe_layer:
                new_past.append(
                    (
                        past_key_values[idx][0][:, :, :length_after_fastv, :],
                        past_key_values[idx][1][:, :, :length_after_fastv, :],
                    )
                )
            else:
                new_past.append(
                    (
                        past_key_values[idx][0][:, :, :length_after_fastv_2, :],
                        past_key_values[idx][1][:, :, :length_after_fastv_2, :],
                    )
                )
    
    past_key_values = tuple(new_past)
    return past_key_values

