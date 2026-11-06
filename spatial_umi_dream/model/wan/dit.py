"""Causal Wan DiT with action registers and optional state registers.

Training is teacher forcing on the token sequence
``[clean video | noisy video | action registers | state registers]``: the clean
half is block-causal, each noisy video/action block attends to the clean
prefix, its own block and its register block. Inference runs one latent frame
per call against a per-block KV cache of roped keys/values.

State-dict layout and parameter creation order are shared with the released
checkpoints; do not rename or reorder submodules.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_umi_dream.model.wan.attention import AttentionModule, flash_attention


# ---------------------------------------------------------------------------
# Embeddings and RoPE
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Complex (polar) rotary frequencies, shape [max_seq_len, dim // 2]."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    freqs_state: torch.Tensor | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    """Teacher-forcing RoPE: 3D video freqs followed by 1D register freqs.

    With ``action_register_length`` the register layout is
    ``[action (chunks * num_action_per_block) | state (chunks * num_state_per_block)]``
    and each part gets its own 1D frequency table starting at position 0.
    """
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None
        use_state = freqs_state is not None and num_state_per_block is not None and num_state_per_block > 0
        if use_state:
            chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
            freqs_1d_action = freqs_action[:chunk_size * num_action_per_block].view(
                chunk_size * num_action_per_block, 1, -1,
            )
            freqs_1d_state = freqs_state[:chunk_size * num_state_per_block].view(
                chunk_size * num_state_per_block, 1, -1,
            )
            freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)
        else:
            chunk_size = action_register_length // num_action_per_block
            freqs_1d_action = freqs_action[:chunk_size * num_action_per_block].view(
                chunk_size * num_action_per_block, 1, -1,
            )
            freqs = torch.cat([freqs, freqs_1d_action], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    action_state_index: int,
    freqs_state: torch.Tensor | None = None,
    num_state_per_block: int | None = None,
):
    """Cached-inference RoPE for one block: the register freqs are the slice for
    block ``action_state_index`` so they line up with the teacher-forcing positions."""
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        use_state = (
            freqs_state is not None
            and num_state_per_block is not None
            and num_state_per_block > 0
        )
        if use_state:
            assert action_register_length == num_action_per_block + num_state_per_block
        else:
            assert action_register_length == num_action_per_block

        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        if use_state:
            freqs_state_slice = freqs_state[
                action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
            ]
            freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
                action_register_length, 1, -1,
            )
        else:
            freqs_1d = freqs_action_slice.view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


# ---------------------------------------------------------------------------
# Action / state encoders (Pi0-style, per-embodiment weights)
# ---------------------------------------------------------------------------

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, w) for timesteps of shape (B, T)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B, T)  -- one timestep per action token
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


# ---------------------------------------------------------------------------
# Wan building blocks
# ---------------------------------------------------------------------------

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)


class WanI2VCrossAttention(nn.Module):
    """Cross-attention to text (umt5) plus CLIP image tokens (first 257 context tokens)."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, 257 + L2, C]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        # KV cache length kept at inference: the local window, or 21 frames for global attention.
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        # 0 when state is disabled. Non-zero unlocks the three-way register split
        # (image | action | state) in forward() and `_process_state_blocks`.
        self.num_state_per_block = num_state_per_block
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule()
        self.causal_attn = AttentionModule(causal=True)

    def _process_clean_image_only(self, clean_image_q, clean_image_k, clean_image_v, clean_frames):
        """Block-causal attention over the clean video half.

        First frame: conditioning, self-attention only.
        Block i: attends to the first frame + previous blocks within the local
        window + itself.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block

        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)

        # First frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            clean_image_k[:, :self.frame_seqlen],
            clean_image_v[:, :self.frame_seqlen]
        )

        if self.local_attn_size == -1:
            # Global attention: one causal call, each token sees the first frame
            # and everything up to itself.
            blocks_q = clean_image_q[:, self.frame_seqlen:]
            blocks_k = clean_image_k
            blocks_v = clean_image_v

            output[:, self.frame_seqlen:] = self.causal_attn(
                blocks_q, blocks_k, blocks_v
            )
        else:
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]

            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]

                q_block = clean_image_q[:, block_start:block_end]

                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_context = torch.cat([
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end]  # Recent blocks + current
                ], dim=1)
                v_context = torch.cat([
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end]
                ], dim=1)

                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                    clean_image_k, clean_image_v,
                                    noisy_action_k, noisy_action_v,
                                    half_frames,
                                    noisy_state_k=None, noisy_state_v=None):
        """Noisy video blocks under teacher forcing.

        First noisy frame: conditioning, self-attend only.
        Block i: attends to first_clean_frame + clean_blocks[0:i] +
        current_noisy_block + action[i] [+ state[i]].
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        output = torch.empty_like(noisy_image_q)

        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen]
        )

        use_state = noisy_state_k is not None

        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        if use_state:
            state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
            state_block_ends = [start + self.num_state_per_block for start in state_block_starts]

        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]

            q_block = noisy_image_q[:, noisy_start:noisy_end]

            k_parts = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
            ]
            v_parts = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
            ]
            if use_state:
                k_parts.append(noisy_state_k[:, state_block_starts[block_idx]:state_block_ends[block_idx]])
                v_parts.append(noisy_state_v[:, state_block_starts[block_idx]:state_block_ends[block_idx]])
            k_context = torch.cat(k_parts, dim=1)
            v_context = torch.cat(v_parts, dim=1)

            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                     clean_image_k, clean_image_v,
                                     noisy_image_k, noisy_image_v,
                                     half_frames,
                                     noisy_state_k=None, noisy_state_v=None):
        """Action register blocks under teacher forcing.

        Action block i: attends to first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] [+ state[i]].
        """
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        output = torch.empty_like(noisy_action_q)

        use_state = noisy_state_k is not None

        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_ends = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_block_starts]
        if use_state:
            state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
            state_block_ends = [start + self.num_state_per_block for start in state_block_starts]

        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            noisy_img_start = noisy_image_block_starts[block_idx]
            noisy_img_end = noisy_image_block_ends[block_idx]

            q_block = noisy_action_q[:, action_start:action_end]

            k_parts = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_img_start:noisy_img_end],
                noisy_action_k[:, action_start:action_end],
            ]
            v_parts = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
            ]
            if use_state:
                k_parts.append(noisy_state_k[:, state_block_starts[block_idx]:state_block_ends[block_idx]])
                v_parts.append(noisy_state_v[:, state_block_starts[block_idx]:state_block_ends[block_idx]])
            k_context = torch.cat(k_parts, dim=1)
            v_context = torch.cat(v_parts, dim=1)

            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """Block-diagonal self-attention for state tokens.

        State is pure conditioning written into image/action K/V downstream;
        its own Q row only self-attends within its block. With
        `num_state_per_block=1` each call is a single-token softmax (==1), so
        the output reduces to V.
        """
        assert self.num_state_per_block > 0
        num_blocks = state_horizon // self.num_state_per_block

        if num_blocks == 1:
            return self.attn(state_q, state_k, state_v)

        output = torch.empty_like(state_q)
        for block_idx in range(num_blocks):
            sb = block_idx * self.num_state_per_block
            se = sb + self.num_state_per_block
            output[:, sb:se] = self.attn(
                state_q[:, sb:se],
                state_k[:, sb:se],
                state_v[:, sb:se],
            )
        return output

    def _forward_teacher_forcing(self, q, k, v, s, freqs, freqs_action, action_register_length, state_kwargs):
        """Sequence layout: [clean video (half) | noisy video (half) | action | state]."""
        if action_register_length is None:
            raise ValueError("Teacher-forcing training requires action registers (video-only training is not supported).")
        q_context = q[:, :(s-action_register_length)//2]
        k_context = k[:, :(s-action_register_length)//2]
        q_noisy = q[:, (s-action_register_length)//2:]
        k_noisy = k[:, (s-action_register_length)//2:]

        # rope should be same for clean and noisy parts
        rq_context = rope_action_apply(
            x=q_context,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=None,
        ).type_as(v)
        rk_context = rope_action_apply(
            x=k_context,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=None,
        ).type_as(v)

        rq_noisy = rope_action_apply(
            x=q_noisy,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            **state_kwargs,
        ).type_as(v)
        rk_noisy = rope_action_apply(
            x=k_noisy,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            **state_kwargs,
        ).type_as(v)

        roped_query = torch.cat([rq_context, rq_noisy], dim=1)
        roped_key = torch.cat([rk_context, rk_noisy], dim=1)

        # Clean half: [image tokens only]
        # Noisy half: [image tokens][action tokens][state tokens (optional)]
        # Causality only applies to image blocks.
        half_seq_len = (s - action_register_length) // 2
        clean_image_seq_len = half_seq_len
        clean_frames = clean_image_seq_len // self.frame_seqlen

        noisy_image_seq_len = half_seq_len
        noisy_frames = noisy_image_seq_len // self.frame_seqlen
        num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
        action_horizon = num_image_blocks * self.num_action_per_block

        use_state = bool(state_kwargs)
        state_horizon = num_image_blocks * self.num_state_per_block if use_state else 0

        expected_len = half_seq_len + noisy_image_seq_len + action_horizon + state_horizon
        if roped_query.shape[1] != expected_len:
            raise ValueError(
                "Sequence length does not match block layout. "
                "For 5B use 320x176 (e.g. image_resolution_width=320, image_resolution_height=176). "
                f"Got noisy_frames={noisy_frames}, num_image_blocks={num_image_blocks}, "
                f"action_register_length={action_register_length}, "
                f"use_state={use_state}, state_horizon={state_horizon}. "
                "Ensure (noisy_frames - 1) // num_frame_per_block >= 1 and register length equals "
                "num_blocks * (num_action_per_block + num_state_per_block)."
            )

        clean_image_q = roped_query[:, :clean_image_seq_len]
        clean_image_k = roped_key[:, :clean_image_seq_len]
        clean_image_v = v[:, :clean_image_seq_len]

        noisy_image_q = roped_query[:, half_seq_len:half_seq_len + noisy_image_seq_len]
        noisy_action_q = roped_query[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]

        noisy_image_k = roped_key[:, half_seq_len:half_seq_len + noisy_image_seq_len]
        noisy_action_k = roped_key[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]

        noisy_image_v = v[:, half_seq_len:half_seq_len + noisy_image_seq_len]
        noisy_action_v = v[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]

        if use_state:
            state_offset = half_seq_len + noisy_image_seq_len + action_horizon
            noisy_state_q = roped_query[:, state_offset:state_offset + state_horizon]
            noisy_state_k = roped_key[:, state_offset:state_offset + state_horizon]
            noisy_state_v = v[:, state_offset:state_offset + state_horizon]
        else:
            noisy_state_q = noisy_state_k = noisy_state_v = None

        clean_image_outputs = self._process_clean_image_only(
            clean_image_q, clean_image_k, clean_image_v, clean_frames)

        noisy_image_outputs = self._process_noisy_image_blocks(
            noisy_image_q, noisy_image_k, noisy_image_v,
            clean_image_k, clean_image_v,
            noisy_action_k, noisy_action_v,
            noisy_frames,
            noisy_state_k=noisy_state_k,
            noisy_state_v=noisy_state_v)

        noisy_action_outputs = self._process_noisy_action_blocks(
            noisy_action_q, noisy_action_k, noisy_action_v,
            clean_image_k, clean_image_v,
            noisy_image_k, noisy_image_v,
            noisy_frames,
            noisy_state_k=noisy_state_k,
            noisy_state_v=noisy_state_v)

        if use_state:
            noisy_state_outputs = self._process_state_blocks(
                noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)
            x = torch.cat([
                clean_image_outputs,
                noisy_image_outputs, noisy_action_outputs, noisy_state_outputs,
            ], dim=1)
        else:
            x = torch.cat([
                clean_image_outputs,
                noisy_image_outputs, noisy_action_outputs,
            ], dim=1)
        return x

    def _forward_cached(self, q, k, v, freqs, freqs_action, action_register_length, kv_cache,
                        current_start_frame, state_kwargs):
        """One new frame block (plus its register) against the roped KV cache.

        Returns the attention output and the updated cache ``[2, B, L, n, d]``,
        truncated to ``max_attention_size`` for local attention.
        """
        action_state_index = (current_start_frame - 1) // self.num_frame_per_block

        roped_query = causal_rope_action_apply(
            x=q,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            action_state_index=action_state_index,
            **state_kwargs,
        ).type_as(v)
        roped_key = causal_rope_action_apply(
            x=k,
            freqs=freqs,
            freqs_action=freqs_action,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            action_state_index=action_state_index,
            **state_kwargs,
        ).type_as(v)

        # split roped_query and roped_action_query (the last action_register_length tokens)
        roped_action_query: torch.Tensor | None = None
        roped_action_key: torch.Tensor | None = None
        action_v: torch.Tensor | None = None

        if action_register_length is not None:
            roped_action_query = roped_query[:, -action_register_length:]
            roped_query = roped_query[:, :-action_register_length]
            roped_action_key = roped_key[:, -action_register_length:]
            roped_key = roped_key[:, :-action_register_length]
            action_v = v[:, -action_register_length:]
            v = v[:, :-action_register_length]

        # Append the new video keys/values; registers never enter the cache.
        new_k = torch.cat([kv_cache[0], roped_key], dim=1)
        new_v = torch.cat([kv_cache[1], v], dim=1)

        # We may need to truncate the KV cache if it's size is larger than the max attention size.
        new_k = new_k[:, -self.max_attention_size:]
        new_v = new_v[:, -self.max_attention_size:]

        if action_register_length is not None:
            # Inference register layout (per block, set in _forward_blocks):
            # [action(num_action_per_block) | state(num_state_per_block?)].
            # When state is enabled, split state_q out and run it through a
            # dedicated self-only attention to match training's
            # `_process_state_blocks`. Otherwise state_q would attend to the
            # full image cache + action register, diverging from training.
            if state_kwargs:
                state_len = self.num_state_per_block
                action_len = action_register_length - state_len
                assert action_len == self.num_action_per_block
                action_q_ = roped_action_query[:, :action_len]
                state_q_ = roped_action_query[:, action_len:]
                action_k_ = roped_action_key[:, :action_len]
                state_k_ = roped_action_key[:, action_len:]
                action_v_ = action_v[:, :action_len]
                state_v_ = action_v[:, action_len:]

                x_main = self.attn(
                    torch.cat([roped_query, action_q_], dim=1),
                    torch.cat([new_k, action_k_, state_k_], dim=1),
                    torch.cat([new_v, action_v_, state_v_], dim=1),
                )
                x_state = self.attn(state_q_, state_k_, state_v_)
                x = torch.cat([x_main, x_state], dim=1)
            else:
                x = self.attn(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([new_k, roped_action_key], dim=1),
                    torch.cat([new_v, action_v], dim=1),
                )
        else:
            x = self.attn(
                roped_query,
                new_k,
                new_v,
            )
        updated_kv_cache = torch.stack([new_k, new_v], dim=0)
        return x, updated_kv_cache

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        freqs_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            freqs(Tensor): Rope freqs for the video tokens, shape [L_video, 1, C / num_heads / 2]
            kv_cache(Tensor): [2, B, L_cache, num_heads, head_dim] at inference; None for teacher forcing.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        updated_kv_cache: torch.Tensor | None = None

        # When state is enabled the register layout is [action | state] per block;
        # rope helpers receive freqs_state + num_state_per_block to interleave the
        # state frequency slice into the rotation. num_state_per_block=0 keeps
        # callers in the no-state code path.
        state_kwargs: dict[str, object] = {}
        if freqs_state is not None and self.num_state_per_block > 0:
            state_kwargs = dict(
                freqs_state=freqs_state,
                num_state_per_block=self.num_state_per_block,
            )

        if kv_cache is None:
            x = self._forward_teacher_forcing(
                q, k, v, s, freqs, freqs_action, action_register_length, state_kwargs)
        else:
            x, updated_kv_cache = self._forward_cached(
                q, k, v, freqs, freqs_action, action_register_length, kv_cache,
                current_start_frame, state_kwargs)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=0):
        super().__init__()

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        freqs_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, 6, C] (one modulation vector per token)
            freqs(Tensor): Rope freqs, shape [L_video, 1, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(
            (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        )
        x = x + (y * e[5].squeeze(2))
        return x, updated_kv_cache


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, 1, C]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalWanModel(nn.Module):
    r"""
    Causal Wan diffusion backbone (image-to-video) with action/state registers.
    Defaults are the Wan2.2-TI2V-5B dimensions.
    """

    def __init__(self,
                 frame_seqlen,
                 max_chunk_size=-1,
                 num_frame_per_block=1,
                 num_action_per_block=32,
                 action_dim=32,
                 use_state=False,
                 max_state_dim=32,
                 num_state_per_block=1,
                 in_dim=48,
                 dim=3072,
                 ffn_dim=14336,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=48,
                 num_heads=24,
                 num_layers=30,
                 eps=1e-6,
                 hidden_size=1024,
                 text_len=512,
                 patch_size=(1, 2, 2),
                 qk_norm=True,
                 cross_attn_norm=True):
        r"""
        Args:
            frame_seqlen: video tokens per latent frame after patching.
            max_chunk_size: local attention window in blocks (-1 = global attention).
            num_frame_per_block: latent frames per causal block.
            num_action_per_block / action_dim: action tokens per block and their width.
            use_state / max_state_dim / num_state_per_block: optional proprio state registers.
            in_dim / out_dim: latent channels; dim / ffn_dim / num_heads / num_layers / eps: transformer size.
            freq_dim: sinusoidal timestep embedding width; text_dim / text_len: umt5 context.
            hidden_size: hidden width of the action decoder / state encoder MLPs.
        """
        super().__init__()


        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.num_frame_per_block = num_frame_per_block
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        # State register switch. When false, no state_encoder / freqs_state are
        # constructed so the state-dict round-trips with the state-free baseline
        # checkpoint. The CausalWanSelfAttention layers receive num_state_per_block=0
        # in this case, falling through to the original code paths.
        self.use_state = use_state
        self.max_state_dim = max_state_dim
        self.num_state_per_block = num_state_per_block if use_state else 0

        # All released checkpoints were trained with a single embodiment slot.
        max_num_embodiments = 1

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )
        if self.use_state:
            self.state_encoder = CategorySpecificMLP(
                num_categories=max_num_embodiments,
                input_dim=max_state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.dim,
            )
        else:
            self.state_encoder = None

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads, frame_seqlen,
                                    self.local_attn_size, num_frame_per_block, qk_norm, cross_attn_norm, eps,
                                    num_action_per_block,
                                    self.num_state_per_block)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        self.freqs_action = rope_params(1024*10, d)
        # State 1D RoPE only allocated when state is enabled. Same head_dim as
        # action so the rope helpers can concat freqs_action and freqs_state
        # along the register dimension.
        if self.use_state:
            self.freqs_state = rope_params(1024, d)
        else:
            self.freqs_state = None
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = True

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Cached-inference pass through the blocks for one latent frame
        (this is the method the server wraps in torch.compile).
        """
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        num_t = timestep.shape[1]

        use_state = self.use_state and state is not None and action is not None
        state_length = 0

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            if use_state:
                state_features = self.state_encoder(state, embodiment_id)
                state_length = state_features.shape[1]
                # Register layout: [action_features (action_length), state_features (state_length)]
                action_register = torch.cat([action_features, state_features], dim=1)
            else:
                action_register = action_features
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = 0
            action_register_length = None

        # time embeddings: expand to exactly seq_len so e matches x (5B: frame_seqlen=50, 1 frame -> 50 tokens)
        repeat = (seq_len + num_t - 1) // num_t
        timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]

        if action is not None:
            assert timestep_action is not None
            if use_state:
                # Each state token's timestep = the corresponding action chunk's
                # first timestep (stride = action_tokens_per_state).
                stride = timestep_action.shape[1] // state_length
                timestep_state = timestep_action[:, ::stride]
                timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)
            else:
                timestep = torch.cat([timestep, timestep_action], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            # Pull action tokens out of the register slab. State tokens (if any)
            # live at [seq_len + action_length : seq_len + action_length + state_length]
            # and are discarded (state is conditioning-only by design).
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Video tokens only
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches

    def forward_cached(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Run the diffusion model with kv caching for one latent frame
        (Algorithm 2 of CausVid, https://arxiv.org/abs/2412.07772).

        Args:
            x (Tensor): noisy latent [B, C_in, F, H, W]
            timestep (Tensor): [B, F]
            context (Tensor): text embeddings [B, text_len, C]
            seq_len (`int`): number of video tokens in x
            kv_cache (list[Tensor]): per-block [2, B, L_cache, n, d] roped K/V
            action / timestep_action / state: action registers (None for the clean first-frame pass)
            clip_feature (Tensor, *optional*): CLIP image features
        Returns:
            (video_noise_pred [B, C_out, F, H, W], action_noise_pred or None, updated kv caches)
        """
        assert context.shape[1] == self.text_len

        # embeddings
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        # Detach outputs from the (possibly CUDA-graph) compiled region's buffers.
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def forward(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x,
        clip_feature=None,
        action=None,
        state=None,
    ):
        r"""
        Teacher-forcing training pass.

        Args:
            x (Tensor): noisy latents [B, C_in, F, H, W]
            timestep (Tensor): [B, F] per-frame video timesteps
            timestep_action (Tensor): [B, H] per-token action timesteps
            context (Tensor): text embeddings [B, text_len, C]
            seq_len (`int`): number of video tokens in x
            clean_x (Tensor): clean latents, same shape as x (the teacher-forcing prefix)
            clip_feature (Tensor): CLIP image features
            action (Tensor): noisy actions [B, H, action_dim]
            state (Tensor, *optional*): state tokens [B, S, max_state_dim]
        Returns:
            (video_noise_pred [B, C_out, F, H, W], action_noise_pred [B, H, action_dim])
        """
        if action is None:
            raise ValueError("Training requires actions (video-only training is not supported).")

        # embeddings
        x = self.patch_embedding(x)

        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        num_t = timestep.shape[1]

        use_state = self.use_state and state is not None
        state_length = 0

        embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        action_length = action_features.shape[1]
        if use_state:
            state_features = self.state_encoder(state, embodiment_id)
            state_length = state_features.shape[1]
            action_register = torch.cat([action_features, state_features], dim=1)
        else:
            action_register = action_features
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, num_t, seq_len // num_t).reshape(B, -1)
        timestep_original = timestep.clone()

        assert timestep_action is not None
        if use_state:
            stride = timestep_action.shape[1] // state_length
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)
        else:
            timestep = torch.cat([timestep, timestep_action], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Teacher-forcing prefix: clean latents at timestep 0.
        clean_x = self.patch_embedding(clean_x)
        clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
        assert clean_x.shape[1] == seq_len

        x = torch.cat([clean_x, x], dim=1)

        aug_t = torch.zeros_like(timestep_original)

        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
        e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
        e0_clean = self.time_projection(e_clean)
        e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, **kwargs)

        x = x[:, clean_x.shape[1]:]

        action_noise_pred = x[:, seq_len: seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)

        # Video tokens only
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        """3D RoPE table for the video tokens of frames [start_frame, start_frame + f)."""
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state is not None and self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1
        ).reshape(f * h * w, 1, -1)

        return freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
