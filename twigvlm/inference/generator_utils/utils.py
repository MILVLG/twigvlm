from typing import List, Optional, Tuple, Union

import torch
import transformers
from .generator_base import ForwardResult

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

def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
    wipe_layer: Optional[List[int]] = None,
    attention_rank: Optional[List[int]] = None,
    prefill_length: Optional[int] = None, 
    select_indices: Optional[torch.Tensor] = None,
    enable_pruning: Optional[bool] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if enable_pruning is False: # no fastv
        for idx in range(len(past_key_values)):
            merged_kv_0, merged_kv_1 = past_key_values[idx][0][:, :, :maximum_length, :], past_key_values[idx][1][:, :, :maximum_length, :]
            if select_indices is not None:
                merged_kv_0 = torch.cat([merged_kv_0, past_key_values[idx][0][:, :, select_indices+maximum_length, :]], dim=2)
                merged_kv_1 = torch.cat([merged_kv_1, past_key_values[idx][1][:, :, select_indices+maximum_length, :]], dim=2)
            new_past.append(
                (
                    merged_kv_0,
                    merged_kv_1,
                )
            )
    else: # fastv
        image_token_num = maximum_length
        for idx in range(len(past_key_values)):
            if idx in wipe_layer:
                image_token_num = prefill_length + attention_rank[wipe_layer.index(idx)] - attention_rank[-1]
            merged_kv_0, merged_kv_1 = past_key_values[idx][0][:, :, :image_token_num, :], past_key_values[idx][1][:, :, :image_token_num, :]
            if select_indices is not None:
                merged_kv_0 = torch.cat([merged_kv_0, past_key_values[idx][0][:, :, select_indices+image_token_num, :]], dim=2)
                merged_kv_1 = torch.cat([merged_kv_1, past_key_values[idx][1][:, :, select_indices+image_token_num, :]], dim=2)
            new_past.append(
                (
                    merged_kv_0,
                    merged_kv_1,
                )
            )
    
    past_key_values = tuple(new_past)
    return past_key_values


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

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, tree_mask):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if tree_mask is not None:
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        return combined_attention_mask

def find_first_index(vec, x):
    # 创建布尔掩码，标记等于x的位置
    mask = (vec == x)
    # 获取非零索引（所有等于x的位置）
    indices = mask.nonzero()
    
    if indices.size(0) > 0:  # 如果存在至少一个匹配项
        return indices[0].item()  # 返回第一个匹配项的索引
    else:
        return -1  # 未找到返回-1
    
def PHead_attention_module(self, qk, tags):
    q, k, h = qk
    batch_scores = []

    for i in range(h.shape[0]):  # batch dimension
        it = tags[i]
        weight = torch.zeros(h[i].shape[0],device=h.device)
        q_idx = find_first_index(it, -3)  # question end index
        k_mask = (it == 1)                                                     
        qi = q[i, :, q_idx, :].unsqueeze(0)  # question token
        ki = k[i, :, k_mask, :]  # image tokens
        xq = h[i, q_idx, :].unsqueeze(0)
        xk = h[i, k_mask, :]
        prob = self.model.leaf_attention_module(qi, ki, xq, xk)
        weight[k_mask] = prob
        weight = weight * (it == 1)
        batch_scores.append(weight)

    return torch.cat(batch_scores, dim=0).unsqueeze(0)