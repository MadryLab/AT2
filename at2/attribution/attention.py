"""
Utilities for extracting and manipulating attention weights from transformer models,
starting from pre-computed hidden states.

This module provides functions to compute attention weights from various transformer
models (like Llama, Phi, Qwen, Gemma) and use them for attribution. We compute only
the relevant attention weights (as specified by `attribution_start` and
`attribution_end`) in order to be able to efficiently compute and store them. If we
were to use `output_attentions=True` in the forward pass, we would (1) only be able
to use the `eager` attention implementation, and (2) would need to store the entire
attention matrix which grows quadratically with the sequence length. Most of the
logic here is replicated from the `transformers` library.

If you'd like to perform attribution on a model that is not currently supported,
you can add it yourself by modifying `infer_model_type` and
`get_layer_attention_weights`. Please see `tests/attribution/test_attention.py`
to ensure that your implementation matches the expected attention weights when
using the `output_attentions=True`.
"""

import math
from typing import Any, Optional
import torch as ch
import transformers.models


def infer_model_type(model):
    model_type_to_keyword = {
        "llama": "llama",
        "phi3": "phi",
        "qwen2": "qwen",
        "gemma3": "gemma",
    }
    for model_type, keyword in model_type_to_keyword.items():
        if keyword in model.name_or_path.lower():
            return model_type
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}. Specify `model_type`.")


def get_helpers(model_type):
    if not hasattr(transformers.models, model_type):
        raise ValueError(f"Unknown model: {model_type}")
    model_module = getattr(transformers.models, model_type)
    modeling_module = getattr(model_module, f"modeling_{model_type}")
    return modeling_module.apply_rotary_pos_emb, modeling_module.repeat_kv


def get_position_ids_and_attention_mask(model, hidden_states):
    input_embeds = hidden_states[0]
    _, seq_len, _ = input_embeds.shape
    position_ids = ch.arange(0, seq_len, device=input_embeds.device).unsqueeze(0)
    attention_mask = ch.ones(
        seq_len, seq_len + 1, device=input_embeds.device, dtype=model.dtype
    )
    attention_mask = ch.triu(attention_mask, diagonal=1)
    attention_mask *= ch.finfo(model.dtype).min
    attention_mask = attention_mask[None, None]
    return position_ids, attention_mask


def get_attentions_shape(model):
    num_layers = len(model.model.layers)
    num_heads = model.model.config.num_attention_heads
    return num_layers, num_heads


def get_layer_attention_weights(
    model,
    hidden_states,
    layer_index,
    position_ids,
    attention_mask,
    attribution_start=None,
    attribution_end=None,
    model_type=None,
):
    model_type = model_type or infer_model_type(model)
    assert layer_index >= 0 and layer_index < len(model.model.layers)
    layer = model.model.layers[layer_index]
    self_attn = layer.self_attn
    hidden_states = hidden_states[layer_index]
    hidden_states = layer.input_layernorm(hidden_states)
    bsz, q_len, _ = hidden_states.size()

    num_attention_heads = model.model.config.num_attention_heads
    num_key_value_heads = model.model.config.num_key_value_heads
    head_dim = self_attn.head_dim

    if model_type in ("llama", "qwen2", "gemma3"):
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
    elif model_type in ("phi3",):
        qkv = self_attn.qkv_proj(hidden_states)
        query_pos = num_attention_heads * head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}")

    query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
    key_states = key_states.transpose(1, 2)

    if model_type in ("gemma3",):
        query_states = self_attn.q_norm(query_states)
        key_states = self_attn.k_norm(key_states)

        if self_attn.is_sliding:
            position_embeddings = model.model.rotary_emb_local(
                hidden_states, position_ids
            )
        else:
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    else:
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    cos, sin = position_embeddings

    apply_rotary_pos_emb, repeat_kv = get_helpers(model_type)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attribution_start = attribution_start if attribution_start is not None else 1
    attribution_end = attribution_end if attribution_end is not None else q_len + 1
    causal_mask = causal_mask[:, :, attribution_start - 1 : attribution_end - 1]
    query_states = query_states[:, :, attribution_start - 1 : attribution_end - 1]

    attn_weights = ch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        head_dim
    )
    attn_weights = attn_weights + causal_mask
    dtype = attn_weights.dtype
    attn_weights = ch.softmax(attn_weights, dim=-1, dtype=ch.float32).to(dtype)
    return attn_weights


def get_attention_weights(
    model: Any,
    hidden_states: Any,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Any:
    """
    Compute the attention weights for the given model and hidden states.

    Args:
        model: The model to compute the attention weights for.
        hidden_states: The pre-computed hidden states.
        attribution_start: The start index of the tokens we would like to attribute.
        attribution_end: The end index of the tokens we would like to attribute.
        model_type: The type of model to compute the attention weights for (each model
            in the `transformers` library has its own specific attention implementation).
    """
    with ch.no_grad():
        position_ids, attention_mask = get_position_ids_and_attention_mask(
            model, hidden_states
        )
        num_layers, num_heads = get_attentions_shape(model)
        num_tokens = hidden_states[0].shape[1] + 1
        attribution_start = attribution_start if attribution_start is not None else 1
        attribution_end = attribution_end if attribution_end is not None else num_tokens
        num_target_tokens = attribution_end - attribution_start
        weights = ch.zeros(
            num_layers,
            num_heads,
            num_target_tokens,
            num_tokens - 1,
            device=model.device,
            dtype=model.dtype,
        )
        for i in range(len(model.model.layers)):
            cur_weights = get_layer_attention_weights(
                model,
                hidden_states,
                i,
                position_ids,
                attention_mask,
                attribution_start=attribution_start,
                attribution_end=attribution_end,
                model_type=model_type,
            )
            weights[i, :, :, :] = cur_weights[0]
    return weights
