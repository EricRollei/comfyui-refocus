"""
Custom transformer forward pass for Genfocus multi-conditional generation.
Ported from rayray9999/Genfocus

This module provides custom forward functions that allow:
1. Multiple image branches (main + conditions) processed in parallel
2. Per-branch LoRA adapter switching
3. Cross-attention between branches controlled by group_mask
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

try:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from diffusers.models.attention_processor import Attention
    from diffusers.models.embeddings import apply_rotary_emb
except ImportError:
    FluxTransformer2DModel = None
    Attention = None
    apply_rotary_emb = None

try:
    from accelerate.utils import is_torch_version
except ImportError:
    def is_torch_version(op, version):
        return True

from .lora_utils import specify_lora


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    """Clip hidden states to prevent numerical instability."""
    return torch.clamp(hidden_states, min=-1e4, max=1e4)


def attn_forward(
    attn: "Attention",
    hidden_states: List[torch.FloatTensor],
    adapters: List[str],
    hidden_states2: Optional[List[torch.FloatTensor]] = None,
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    to_cache: Optional[List[bool]] = None,
    cache_storage: Optional[List[List[List[torch.Tensor]]]] = None,
    **kwargs: dict,
) -> torch.FloatTensor:
    """
    Custom attention forward that handles multiple branches with different adapters.
    
    Args:
        attn: Attention module
        hidden_states: List of hidden states for image branches
        adapters: List of adapter names, one per branch
        hidden_states2: List of hidden states for text branches (optional)
        position_embs: Rotary position embeddings per branch
        group_mask: Boolean tensor [N, N] controlling cross-attention between branches
        cache_mode: "write" to save KV cache, "read" to use cached values
        to_cache: List of booleans indicating which branches to cache
        cache_storage: Storage for KV cache
    
    Returns:
        Tuple of (image_outputs, text_outputs) or just image_outputs
    """
    if hidden_states2 is None:
        hidden_states2 = []
    
    bs = hidden_states[0].shape[0]
    h2_n = len(hidden_states2)
    
    queries, keys, values = [], [], []
    
    # Process text branches (encoder hidden states)
    for i, hidden_state in enumerate(hidden_states2):
        query = attn.add_q_proj(hidden_state)
        key = attn.add_k_proj(hidden_state)
        value = attn.add_v_proj(hidden_state)
        
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        
        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)
        
        queries.append(query)
        keys.append(key)
        values.append(value)
    
    # Process image branches with per-branch adapters
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)
        
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        
        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)
        
        queries.append(query)
        keys.append(key)
        values.append(value)
    
    # Apply rotary embeddings if provided
    if position_embs is not None:
        queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
        keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]
    
    # Handle KV caching
    if cache_mode == "write" and cache_storage is not None and to_cache is not None:
        for i, (k, v) in enumerate(zip(keys, values)):
            if to_cache[i]:
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)
    
    # Compute attention for each branch
    attn_outputs = []
    for i, query in enumerate(queries):
        keys_, values_ = [], []
        
        # Gather keys/values from other branches based on group_mask
        for j, (k, v) in enumerate(zip(keys, values)):
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
        
        # Add cached keys/values if reading from cache
        if cache_mode == "read" and cache_storage is not None:
            keys_.extend(cache_storage[attn.cache_idx][0])
            values_.extend(cache_storage[attn.cache_idx][1])
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            query, 
            torch.cat(keys_, dim=2), 
            torch.cat(values_, dim=2)
        ).to(query.dtype)
        
        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * (keys_[0].shape[-1]))
        attn_outputs.append(attn_output)
    
    # Apply output projections
    h_out, h2_out = [], []
    
    for i, hidden_state in enumerate(hidden_states2):
        h2_out.append(attn.to_add_out(attn_outputs[i]))
    
    for i, hidden_state in enumerate(hidden_states):
        h = attn_outputs[i + h2_n]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                h = attn.to_out[0](h)
        h_out.append(h)
    
    return (h_out, h2_out) if h2_n else h_out


def block_forward(
    self,
    image_hidden_states: List[torch.FloatTensor],
    text_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward_fn=None,
    **kwargs: dict,
):
    """
    Forward pass for a dual-stream transformer block.
    Handles both text and image branches with separate adapters.
    """
    if attn_forward_fn is None:
        attn_forward_fn = attn_forward
    
    txt_n = len(text_hidden_states)
    
    img_variables, txt_variables = [], []
    
    # Compute normalization for text branches
    for i, text_h in enumerate(text_hidden_states):
        txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))
    
    # Compute normalization for image branches with adapters
    for i, image_h in enumerate(image_hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i + txt_n]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + txt_n]))
    
    # Attention
    img_attn_output, txt_attn_output = attn_forward_fn(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each[0] for each in txt_variables],
        position_embs=position_embs,
        adapters=adapters,
        **kwargs,
    )
    
    # Process text outputs
    text_out = []
    for i in range(len(text_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
        text_h = text_hidden_states[i] + txt_attn_output[i] * gate_msa.unsqueeze(1)
        norm_h = (
            self.norm2_context(text_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        text_h = self.ff_context(norm_h) * gate_mlp.unsqueeze(1) + text_h
        text_out.append(clip_hidden_states(text_h))
    
    # Process image outputs
    image_out = []
    for i in range(len(image_hidden_states)):
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_h = (
            image_hidden_states[i] + img_attn_output[i] * gate_msa.unsqueeze(1)
        ).to(image_hidden_states[i].dtype)
        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[i + txt_n]):
            image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_out.append(clip_hidden_states(image_h))
    
    return image_out, text_out


def single_block_forward(
    self,
    hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward_fn=None,
    **kwargs: dict,
):
    """
    Forward pass for a single-stream transformer block.
    Used in the later layers of FLUX where text and image are combined.
    """
    if attn_forward_fn is None:
        attn_forward_fn = attn_forward
    
    mlp_hidden_states = [None for _ in hidden_states]
    gates = [None for _ in hidden_states]
    
    hidden_state_norm = []
    for i, hidden_state in enumerate(hidden_states):
        with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):
            h_norm, gates[i] = self.norm(hidden_state, emb=tembs[i])
            mlp_hidden_states[i] = self.act_mlp(self.proj_mlp(h_norm))
        hidden_state_norm.append(h_norm)
    
    attn_outputs = attn_forward_fn(
        self.attn, hidden_state_norm, adapters, position_embs=position_embs, **kwargs
    )
    
    h_out = []
    for i in range(len(hidden_states)):
        with specify_lora((self.proj_out,), adapters[i]):
            h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
            h = gates[i].unsqueeze(1) * self.proj_out(h) + hidden_states[i]
            h_out.append(clip_hidden_states(h))
    
    return h_out


def transformer_forward(
    transformer: "FluxTransformer2DModel",
    image_features: List[torch.Tensor],
    text_features: Optional[List[torch.Tensor]] = None,
    img_ids: Optional[List[torch.Tensor]] = None,
    txt_ids: Optional[List[torch.Tensor]] = None,
    pooled_projections: Optional[List[torch.Tensor]] = None,
    timesteps: Optional[List[torch.LongTensor]] = None,
    guidances: Optional[List[torch.Tensor]] = None,
    adapters: Optional[List[str]] = None,
    single_block_forward_fn=None,
    block_forward_fn=None,
    attn_forward_fn=None,
    **kwargs: dict,
):
    """
    Custom transformer forward that processes multiple image and text branches.
    
    This is the core of Genfocus - it allows:
    1. Multiple image conditions to be processed in parallel
    2. Each branch to use a different LoRA adapter
    3. Cross-attention between branches controlled by group_mask
    
    Args:
        transformer: FluxTransformer2DModel instance
        image_features: List of image latent tensors (main + conditions)
        text_features: List of text embeddings
        img_ids: Position IDs for each image branch
        txt_ids: Position IDs for text
        pooled_projections: Pooled text embeddings
        timesteps: Timestep for each branch
        guidances: Guidance scale for each branch
        adapters: LoRA adapter name for each branch
    
    Returns:
        Tuple containing the output noise prediction
    """
    if single_block_forward_fn is None:
        single_block_forward_fn = single_block_forward
    if block_forward_fn is None:
        block_forward_fn = block_forward
    if attn_forward_fn is None:
        attn_forward_fn = attn_forward
    
    self = transformer
    txt_n = len(text_features) if text_features is not None else 0
    
    adapters = adapters or [None] * (txt_n + len(image_features))
    assert len(adapters) == len(timesteps), f"adapters ({len(adapters)}) must match timesteps ({len(timesteps)})"
    
    # Embed image features with per-branch adapters
    image_hidden_states = []
    for i, image_feature in enumerate(image_features):
        with specify_lora((self.x_embedder,), adapters[i + txt_n]):
            image_hidden_states.append(self.x_embedder(image_feature))
    
    # Embed text features
    text_hidden_states = []
    if text_features is not None:
        for text_feature in text_features:
            text_hidden_states.append(self.context_embedder(text_feature))
    
    # Compute timestep embeddings
    assert len(timesteps) == len(image_features) + (len(text_features) if text_features else 0)
    
    def get_temb(timestep, guidance, pooled_projection):
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(image_hidden_states[0].dtype) * 1000
            return self.time_text_embed(timestep, guidance, pooled_projection)
        else:
            return self.time_text_embed(timestep, pooled_projection)
    
    tembs = [get_temb(*each) for each in zip(timesteps, guidances, pooled_projections)]
    
    # Compute position embeddings
    position_embs = [self.pos_embed(each) for each in (*txt_ids, *img_ids)]
    
    # Gradient checkpointing kwargs
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )
    
    # Dual-stream blocks (separate text and image processing)
    for block in self.transformer_blocks:
        block_kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward_fn": attn_forward_fn,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward_fn, **block_kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states, text_hidden_states = block_forward_fn(**block_kwargs)
    
    # Single-stream blocks (combined text and image)
    all_hidden_states = [*text_hidden_states, *image_hidden_states]
    for block in self.single_transformer_blocks:
        block_kwargs = {
            "self": block,
            "hidden_states": all_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward_fn": attn_forward_fn,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            all_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward_fn, **block_kwargs, **gckpt_kwargs
            )
        else:
            all_hidden_states = single_block_forward_fn(**block_kwargs)
    
    # Final output projection (only for main image branch, index txt_n)
    image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
    output = self.proj_out(image_hidden_states)
    
    return (output,)
