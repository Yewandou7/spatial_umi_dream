# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Dense attention kernels used by the causal DiT.

Tensors are [B, L, num_heads, head_dim]. FlashAttention-2 is the default
backend; FlashAttention-3 is used on Hopper when ``flash_attn_interface`` is
installed; PyTorch SDPA is the fallback for GPUs without FlashAttention support
(pre-Ampere) and the CPU test path (``ATTENTION_BACKEND=torch``).
"""

import os
import warnings

import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _gpu_supports_flash_attention():
    """FlashAttention requires Ampere (compute capability 8.0) or newer."""
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False


def _sdpa_attention_fallback(q, k, v, causal=False, dtype=torch.bfloat16):
    """PyTorch SDPA for GPUs that don't support FlashAttention (e.g. pre-Ampere)."""
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=0.
    )
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
):
    """
    q:              [B, Lq, N, C1].
    k:              [B, Lk, N, C1].
    v:              [B, Lk, N, C2].
    causal:         bool. Whether to apply causal attention mask.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        2 for flash attention 2, 3 for flash attention 3, None for 3 if available else 2.

    Returns:
        x:              [B, Lq, N, C2].
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    if not _gpu_supports_flash_attention():
        return _sdpa_attention_fallback(q, k, v, causal=causal, dtype=dtype)

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    q = half(q.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)

    # preprocess key, value
    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))
    k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )
    zeros = torch.zeros([1], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # apply attention
    if version != 2 and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=None,
            causal=causal,
            deterministic=False)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE, "flash_attn (FlashAttention-2) is required on this GPU"
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=0.,
            softmax_scale=None,
            causal=causal,
            window_size=(-1, -1),
            deterministic=False).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


class AttentionModule(torch.nn.Module):
    """Attention over [B, L, N, D] tensors with the backend chosen by ``ATTENTION_BACKEND``.

    ``FA2`` (default) / ``FA3`` go through :func:`flash_attention`; ``torch`` uses
    PyTorch SDPA directly, which also runs on CPU (used by the unit tests).
    """

    def __init__(self, causal: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        backend = os.getenv("ATTENTION_BACKEND", "FA2")
        assert backend in ["torch", "FA2", "FA3"], f"Invalid ATTENTION_BACKEND: {backend}"
        self.backend = backend
        self.causal = causal
        self.compute_dtype = dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.backend == "torch":
            out_dtype = q.dtype
            q = q.transpose(1, 2).to(self.compute_dtype)
            k = k.transpose(1, 2).to(self.compute_dtype)
            v = v.transpose(1, 2).to(self.compute_dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=self.causal,
                dropout_p=0.,
                scale=None,
            )
            out = out.transpose(1, 2).contiguous()
            return out.to(out_dtype)
        return flash_attention(
            q=q, k=k, v=v,
            causal=self.causal,
            dtype=self.compute_dtype,
            version=3 if self.backend == "FA3" else 2,
        )
