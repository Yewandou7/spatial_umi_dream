"""
Vision tower of open-clip ``xlm-roberta-large-vit-huge-14`` (concise re-implementation of
``https://github.com/openai/CLIP`` / ``https://github.com/mlfoundations/open_clip``).

Wan conditions the DiT on the penultimate-layer patch tokens of this ViT-H/14. Only the
visual tower is instantiated; the checkpoint's ``textual.*`` keys are ignored at load time
(``load_state_dict(..., strict=False)``). Module names must match the Wan .pth and our
checkpoints (``image_encoder.model.visual...``, ``image_encoder.model.log_scale``).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange


def _attention(q, k, v, num_heads):
    """Multi-head SDPA on [B, L, C] tensors (what Wan's ``flash_attention`` runs in
    compatibility mode, which is how this encoder always called it)."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x:   [B, L, C].
        """
        # compute query, key, value
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # compute attention
        x = _attention(q, k, v, num_heads=self.num_heads)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert activation in ['quick_gelu', 'gelu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings ('token' pooling: a learned cls token is prepended to the patches)
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # head: unused by ``use_31_block`` but present in the checkpoint
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x, use_31_block=False):
        b = x.size(0)

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1).to(dtype=x.dtype, device=x.device), x], dim=1)
        e = self.pos_embedding
        e = e.to(dtype=x.dtype, device=x.device)
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer; Wan uses the tokens of the penultimate block (31 of 32)
        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class XLMRobertaCLIP(nn.Module):
    """open-clip ViT-H/14 image tower. Named after the full checkpoint; the XLM-RoBERTa
    text tower is not built (its ``textual.*`` weights are skipped when loading)."""

    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 activation='gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))


class WanImageEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # init model
        with torch.device("cpu"):
            self.model = XLMRobertaCLIP()
        # CLIP normalization; only the Normalize step is applied at runtime, resizing is
        # done with F.interpolate on the batched video in encode_image.
        self.transforms = T.Compose([
            T.Resize((self.model.image_size, self.model.image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def encode_image(self, videos):
        """videos: [B, T, 3, H, W] in [-1, 1] -> [B*T, 257, 1280] penultimate-block tokens."""
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u,
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        dtype = next(iter(self.model.visual.parameters())).dtype
        videos = videos.to(dtype)
        out = self.model.visual(videos, use_31_block=True)
        # The outputs of torch compile always need to be cloned before being used.
        out = out.clone()
        return out
