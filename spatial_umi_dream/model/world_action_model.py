"""World action model: a causal Wan2.2 video DiT that jointly denoises the next video block and the action
chunk, conditioned on umt5 text, a CLIP image embedding of the first frame and (optionally) proprio state.

Training is teacher forcing on clean past latents. Inference streams block by block against a KV cache
(`predict_chunk`); with `replay_kv_cache` the cache is rebuilt from buffered recent frames once the local
attention window is full instead of restarting the stream.
"""
import json
import os
import time

from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from torchvision.transforms import v2
import tree
from transformers import PretrainedConfig, PreTrainedModel

from spatial_umi_dream.model.schedulers.flow_match import FlowMatchScheduler
from spatial_umi_dream.model.schedulers.unipc import FlowUniPCMultistepScheduler
from spatial_umi_dream.model.wan.dit import CausalWanModel
from spatial_umi_dream.model.wan.image_encoder import WanImageEncoder
from spatial_umi_dream.model.wan.text_encoder import WanTextEncoder
from spatial_umi_dream.model.wan.vae import WanVAE

# umt5-xxl text encoder and CLIP image encoder weights are shipped with Wan2.1; VAE and DiT with Wan2.2.
WAN21_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"

# DiT kwargs that the config exposes (everything else uses the Wan2.2-TI2V-5B defaults of CausalWanModel).
DIT_CONFIG_KEYS = (
    "frame_seqlen", "max_chunk_size", "num_frame_per_block", "num_action_per_block", "action_dim",
    "use_state", "max_state_dim", "num_state_per_block", "dim", "ffn_dim", "num_heads", "num_layers",
)


def ensure_file(path: str | None, hf_filename: str, repo_id: str) -> str:
    """Return `path` if it exists, otherwise fetch `hf_filename` from the HuggingFace hub. Checkpoint configs
    carry the training machine's paths, which usually do not exist on the serving machine."""
    if path is not None and os.path.exists(path):
        return path
    return hf_hub_download(repo_id=repo_id, filename=hf_filename)


class WorldActionModelConfig(PretrainedConfig):
    model_type = "world_action_model"

    def __init__(
        self,
        # pretrained Wan components
        wan_dit_path: str | None = None,
        text_encoder_path: str | None = None,
        image_encoder_path: str | None = None,
        vae_path: str | None = None,
        load_wan_dit: bool = True,
        # causal DiT
        frame_seqlen: int = 100,
        max_chunk_size: int = 4,
        num_frame_per_block: int = 2,
        num_action_per_block: int = 24,
        action_dim: int = 32,
        action_horizon: int = 24,
        use_state: bool = False,
        max_state_dim: int = 32,
        num_state_per_block: int = 1,
        dim: int = 3072,
        ffn_dim: int = 14336,
        num_heads: int = 24,
        num_layers: int = 30,
        # input video canvas (H, W) the VAE encodes
        target_video_height: int = 320,
        target_video_width: int = 320,
        # dtype of the frozen-module compute / autocast in training
        model_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wan_dit_path = wan_dit_path
        self.text_encoder_path = text_encoder_path
        self.image_encoder_path = image_encoder_path
        self.vae_path = vae_path
        self.load_wan_dit = load_wan_dit
        self.frame_seqlen = frame_seqlen
        self.max_chunk_size = max_chunk_size
        self.num_frame_per_block = num_frame_per_block
        self.num_action_per_block = num_action_per_block
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_state = use_state
        self.max_state_dim = max_state_dim
        self.num_state_per_block = num_state_per_block
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.target_video_height = target_video_height
        self.target_video_width = target_video_width
        self.model_dtype = model_dtype


class WorldActionModel(PreTrainedModel):
    config_class = WorldActionModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: WorldActionModelConfig):
        super().__init__(config)
        self.num_frame_per_block = config.num_frame_per_block
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.use_state = config.use_state

        # Submodule creation order (text encoder, image encoder, VAE, DiT) fixes the random init of the
        # DiT parts that are not in the Wan release; keep it.
        self.text_encoder = WanTextEncoder()
        self.image_encoder = WanImageEncoder()
        self.vae = WanVAE(z_dim=48, dim=160)
        # ZeRO-3 deep-copies list args (feat_cache/feat_idx) when entering each VAE
        # sub-module forward, breaking the causal cache across temporal chunks. Mark the
        # VAE as a z3 leaf so sub-module hooks are bypassed; pair with GatheredParameters
        # in encode_video/encode_image so the (frozen) VAE params are present at call time.
        try:
            from deepspeed.utils.z3_leaf_module import set_z3_leaf_modules
            set_z3_leaf_modules(self, [WanVAE])
        except ImportError:
            pass
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        # Inference settings.
        self.num_inference_steps = 16
        self.seed = 1140
        self.sigma_shift = 5.0
        # RoboTwin server: rebuild the KV cache from buffered frames once the window is full.
        self.replay_kv_cache = False

        # Streaming state shared by the two inference paths: per-layer KV cache, first-frame
        # conditioning, and the position in the latent sequence.
        self.kv_cache = None
        self.clip_feas = None
        self.current_start_frame = 0
        self.language = None
        # (raw_last_frame, vae_latent) per past call so the replay path can rebuild a fresh KV cache from
        # recent frames once `current_start_frame` exceeds `dit.local_attn_size`.
        self._history_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._device = "cuda"

        # NUM_DIT_STEPS: run the DiT on only this many of the 16 UniPC steps; skipped steps reuse the previous
        # flow prediction. The scheduler still takes all 16 steps.
        num_dit_steps = int(os.getenv("NUM_DIT_STEPS", "8"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True] * 16

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.dit = CausalWanModel(**{key: getattr(config, key) for key in DIT_CONFIG_KEYS})

        text_enc_path = ensure_file(config.text_encoder_path, "models_t5_umt5-xxl-enc-bf16.pth", WAN21_HF_REPO_ID)
        self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))
        img_enc_path = ensure_file(
            config.image_encoder_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", WAN21_HF_REPO_ID)
        self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)
        vae_path = ensure_file(config.vae_path, "Wan2.2_VAE.pth", WAN22_HF_REPO_ID)
        self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if config.load_wan_dit:
            self._load_wan_dit_weights(config.wan_dit_path)
        self._noise_logged = False
        self._vae_device_ready = False

        # The DiT is trained in full; the Wan encoders and the VAE stay frozen.
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        num_params = sum(p.numel() for p in self.dit.parameters())
        print(f"DiT parameters: {num_params:,} (all trainable)")

    def _load_wan_dit_weights(self, dit_dir: str | None):
        """Initialise the DiT from the Wan2.2 release (training from Wan). The action/state registers and
        the causal extras are not in the release, so they stay randomly initialised (reported as missing)."""
        if dit_dir is None or not os.path.isdir(dit_dir):
            index_path = hf_hub_download(repo_id=WAN22_HF_REPO_ID, filename="diffusion_pytorch_model.safetensors.index.json")
            dit_dir = os.path.dirname(index_path)
            with open(index_path, 'r') as f:
                index = json.load(f)
            for shard_file in set(index["weight_map"].values()):
                hf_hub_download(repo_id=WAN22_HF_REPO_ID, filename=shard_file)

        index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                shard_files = set(json.load(f)["weight_map"].values())
        else:
            shard_files = {"diffusion_pytorch_model.safetensors"}
        state_dict = {}
        for shard_file in shard_files:
            print(f"Loading Wan2.2 DiT shard: {shard_file}")
            state_dict.update(load_file(os.path.join(dit_dir, shard_file)))
        missing_keys, unexpected_keys = self.dit.load_state_dict(state_dict, strict=False)
        print(f"Loaded Wan2.2 DiT weights from {dit_dir}; randomly initialised (not in the release): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in the Wan2.2 DiT weights: {unexpected_keys}")

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str):
        """Build the model from `checkpoint_dir/config.json` and load its (possibly sharded) safetensors.
        The Wan DiT release is not loaded since the checkpoint overrides every DiT weight."""
        config = WorldActionModelConfig.from_json_file(os.path.join(checkpoint_dir, "config.json"))
        config.load_wan_dit = False
        model = cls(config)
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                shard_files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shard_files = ["model.safetensors"]
        state_dict = {}
        for shard_file in shard_files:
            print(f"Loading checkpoint shard: {shard_file}")
            state_dict.update(load_file(os.path.join(checkpoint_dir, shard_file)))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading the checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading the checkpoint: {unexpected_keys}")
        return model

    def _prepare_inputs(self, inputs: dict) -> dict:
        """Move the collated batch to the model's device; floating tensors take the model dtype."""
        if "action" in inputs:
            action = inputs["action"]
            if action.ndim != 3 or action.shape[1] % self.action_horizon != 0 or action.shape[2] != self.action_dim:
                raise ValueError(f"unexpected action shape {tuple(action.shape)}")

        def to_device(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        return tree.map_structure(to_device, dict(inputs))

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()

    def _create_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Empty per-layer self-attention KV cache."""
        num_heads = self.dit.num_heads
        head_dim = self.dit.dim // num_heads
        return [
            torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device)
            for _ in range(self.dit.num_layers)
        ]

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not self._vae_device_ready:
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def _vae_gather_ctx(self):
        # Under ZeRO-3, VAE params are sharded across ranks. Gather them on every rank
        # before calling self.vae.encode so the conv layers see full weights. No-op when
        # the params are not ZeRO-3 managed.
        try:
            from deepspeed.zero import GatheredParameters
            return GatheredParameters(list(self.vae.parameters()))
        except ImportError:
            from contextlib import nullcontext
            return nullcontext()

    def encode_video(self, input_video):
        self._ensure_vae_on_device(input_video)
        with self._vae_gather_ctx(), torch.no_grad():
            latents = self.vae.encode(input_video)
        return latents

    def encode_image(self, image):
        """First-frame conditioning from image [B, 1, 3, H, W]: CLIP tokens and the VAE latent of the frame.
        (The causal VAE encodes frame 0 on its own, so this equals the first latent of the full clip.)"""
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            self._ensure_vae_on_device(image_input)
            with self._vae_gather_ctx(), torch.no_grad():
                first_latent = self.vae.encode(image_input)
        return clip_context, first_latent

    def _resize_to_target(self, videos: torch.Tensor) -> torch.Tensor:
        """Resize (B, C, T, H, W) video to the configured canvas so the latent token count matches the
        DiT's `frame_seqlen`."""
        target_h, target_w = self.config.target_video_height, self.config.target_video_width
        _, _, _, h, w = videos.shape
        if (h, w) != (target_h, target_w):
            b, c, t, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(b * t, c, h, w),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, c, t, target_h, target_w)
        return videos

    def forward(self, inputs: dict) -> dict:
        """Teacher-forced flow-matching loss on video latents and actions.

        One timestep is drawn per latent frame and shared within each block; the action chunk of a block
        gets the block's timestep. Frame 0 is the clean conditioning frame."""
        data = self._prepare_inputs(inputs)
        self.set_frozen_modules_to_eval_mode()

        has_real_action = data["has_real_action"]
        action_mask = data["action_mask"]

        state_features = data["state"] if self.use_state else None

        actions = data["action"]
        assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        # shape of B * max_length * dim
        prompt_emb = self.encode_prompt(data["text"], data["text_attention_mask"])

        videos = self._resize_to_target(videos)

        latents = self.encode_video(videos)

        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, _ = self.encode_image(image)

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        prompt_emb = prompt_emb.to(self._device)

        noise = torch.randn_like(latents)

        # specific to autoregressive
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        # Uniform timestep per latent frame; frames within a block share the block's first timestep.
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))

        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        noise_action = torch.randn_like(actions)
        assert actions.shape[1] / (noise.shape[1]-1) == (self.dit.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
        if self.use_state and state_features is not None:
            assert (noise.shape[1] - 1) / state_features.shape[1] == (
                self.num_frame_per_block / self.dit.num_state_per_block
            ), (
                f"state_features.shape={state_features.shape}, noise.shape={noise.shape}, "
                f"num_frame_per_block={self.num_frame_per_block}, "
                f"num_state_per_block={self.dit.num_state_per_block}"
            )

        # Action timestep is the timestep of the block the action belongs to.
        timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
        timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)

        if not self._noise_logged:
            video_mean = timestep_id.float().mean().item()
            action_mean = timestep_action_id.float().mean().item()
            print(f"[NOISE] Uniform timesteps | video mean_t={video_mean:.0f} | action (coupled) mean_t={action_mean:.0f}")
            self._noise_logged = True

        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

        timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
        noisy_actions = self.scheduler.add_noise(
            actions.flatten(0, 1),
            noise_action.flatten(0, 1),
            timestep_action.flatten(0, 1),
        ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
        training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            video_noise_pred, action_noise_pred = self.dit(
                noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, context=prompt_emb, seq_len=seq_len,
                action=noisy_actions, timestep_action=timestep_action,
                state=state_features,
                clean_x=latents.transpose(1, 2),
            )

            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            video_mse = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            )  # shape: [B, C, T, H, W]
            dynamics_loss_per_sample = video_mse.mean(dim=(1,3,4))  # shape: [B, T]

            video_time_weight = self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weight_dynamics = dynamics_loss_per_sample * video_time_weight
            weighted_dynamics_loss = weight_dynamics.mean()

            # Logging only: the canvas is [RGB | depth] side by side, so the left half of the latent
            # width is RGB and the right half is depth (spatial). Their mean is the video loss term.
            with torch.no_grad():
                half_width = video_mse.shape[-1] // 2
                rgb_loss = (video_mse[..., :half_width].mean(dim=(1,3,4)) * video_time_weight).mean()
                spatial_loss = (video_mse[..., half_width:].mean(dim=(1,3,4)) * video_time_weight).mean()

            action_loss_per_sample = torch.nn.functional.mse_loss(
                action_noise_pred.float(), training_target_action.float(), reduction='none'
            ) * action_mask  # shape: [B, T_action, D_action]
            # has_real_action is (B,) -> (B, 1, 1) so it broadcasts cleanly against
            # the 3D loss tensor. Without the second None, (B, 1) right-aligns to
            # (1, B, 1) and fails when B != T_action (i.e. any B>1 with multi-chunk
            # action sampling, since T_action = n_chunks * 24 can be 24, 48, 72, 96).
            action_loss_per_sample = has_real_action[:, None, None].float() * action_loss_per_sample
            weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
            weighted_action_loss = weight_action.mean()
            loss = weighted_dynamics_loss + weighted_action_loss

        output_dict = {
            "loss": loss,
            "rgb_loss": rgb_loss,
            "spatial_loss": spatial_loss,
            "action_loss": weighted_action_loss,
        }

        return output_dict

    # ------------------------------------------------------------------ inference

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def _prepare_video(self, videos: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, C) uint8 or float video -> normalised bf16 (B, C, T, H, W) at the DiT's resolution."""
        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        videos = videos.to(dtype=torch.bfloat16)
        return self._resize_to_target(videos)

    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        clip_feature: torch.Tensor,
        kv_cache: list[torch.Tensor],
        kv_cache_metadata: dict[str, bool | int],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One cached DiT call. With `update_kv_cache` the call is a clean-frame prefill that extends the
        cache; otherwise it is a denoising step."""
        obs_noise_pred, action_noise_pred, updated_kv_caches = self.dit.forward_cached(
            noisy_input,
            timestep,
            action=action,
            timestep_action=timestep_action,
            state=state,
            context=context,
            seq_len=seq_len,
            clip_feature=clip_feature,
            kv_cache=kv_cache,
            current_start_frame=kv_cache_metadata["start_frame"],
        )
        if kv_cache_metadata["update_kv_cache"]:
            for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                kv_cache[block_index] = updated_kv_cache.clone()
        obs_noise_pred = obs_noise_pred.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        else:
            action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
        return obs_noise_pred, action_noise_pred

    def _denoise(
        self,
        noise_obs: torch.Tensor,
        noise_action: torch.Tensor,
        prompt_emb: torch.Tensor,
        state_features: torch.Tensor | None,
        seq_len: int,
        kv_cache: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """UniPC-denoise one video block and its action chunk against the prefilled caches, starting at
        `self.current_start_frame`. Returns (video latents, actions, number of DiT steps actually run)."""
        batch_size = noise_obs.shape[0]
        noisy_input = noise_obs
        noisy_input_action = noise_action

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1,
        )
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        dit_compute_steps = 0
        for index in range(len(sample_scheduler.timesteps)):
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]

            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # Steps masked out by NUM_DIT_STEPS reuse the previous step's flow prediction.
            if self.dit_step_mask[index]:
                dit_compute_steps += 1
                flow_pred, flow_pred_action = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    context=prompt_emb,
                    seq_len=seq_len,
                    clip_feature=self.clip_feas,
                    kv_cache=kv_cache,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )

            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                sample=noisy_input,
                step_index=index,
            )
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_action,
                sample=noisy_input_action,
                step_index=index,
            )

        return noisy_input, noisy_input_action, dit_compute_steps

    def _stream_step(self, data: dict) -> dict:
        """Streaming inference for the UMI server.

        The first call of an episode (new prompt or `language is None`) encodes the first
        frame (CLIP + VAE), rebuilds the KV caches and prefills them with the clean frame. Later calls
        VAE-encode the newly observed frames, prefill them, then denoise the next block jointly with the
        action chunk. Once `current_start_frame` reaches `dit.local_attn_size` the stream restarts."""
        start_time = time.perf_counter()

        self.set_frozen_modules_to_eval_mode()
        state_features = data["state"].to(dtype=torch.bfloat16) if self.use_state else None

        videos = self._prepare_video(data["images"])

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif self.current_start_frame >= self.dit.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            self.current_start_frame = 0

        prompt_emb = self.encode_prompt(data["text"], data["text_attention_mask"])

        # A (re)started stream is anchored on the newest observed frame: the window ends at the
        # current time, which is where the predicted action chunk starts (frame 0 in training).
        image = videos[:, :, -1:].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, image = self.encode_image(image)
            self.clip_feas = clip_feas.to(dtype=image.dtype)

        assert self.clip_feas is not None, "the stream has no first-frame conditioning"

        if self.current_start_frame != 0:
            # VAE-encode the 8-frame window observed since the last call. The VAE needs 1 + 4k
            # frames, so the first frame is repeated as the causal anchor.
            videos = torch.cat([videos[:, :, 0:1], videos], dim=2)

            with self._vae_gather_ctx():
                image = self.vae.encode(videos)

        noise_obs = self.generate_noise((image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.dit.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        frame_seqlen = (height // 2) * (width // 2)
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize the KV caches for each new sequence.
            self.kv_cache = self._create_kv_cache(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache is not None
        kv_cache = self.kv_cache

        if self.current_start_frame == 0:
            # Prefill the caches with the clean conditioning frame.
            timestep = torch.zeros([batch_size, 1], device=noise_obs.device, dtype=torch.int64)
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep,
                action=None,
                timestep_action=None,
                context=prompt_emb,
                seq_len=frame_seqlen,
                clip_feature=self.clip_feas,
                kv_cache=kv_cache,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += 1

        if self.current_start_frame != 1:
            # Prefill the caches with the newly observed clean block.
            timestep = torch.zeros([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64)
            current_ref_latents = image[:, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep,
                action=None,
                timestep_action=None,
                context=prompt_emb,
                seq_len=seq_len,
                clip_feature=self.clip_feas,
                kv_cache=kv_cache,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        latents, latents_action, dit_compute_steps = self._denoise(
            noise_obs, noise_action, prompt_emb, state_features, seq_len, kv_cache,
        )

        output = latents
        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        torch.cuda.synchronize()
        print(f"Time taken: Total {time.perf_counter() - start_time:.2f} seconds, DIT Compute Steps {dit_compute_steps} steps")

        return {"action_pred": latents_action, "video_pred": output.transpose(1, 2)}

    def _stream_step_replay(self, data: dict) -> dict:
        """Variant of `_stream_step` that, when `current_start_frame >= dit.local_attn_size`, replays
        recent buffered frames (`self._history_buffer`) to rebuild a fresh KV cache instead of restarting
        the stream, keeping short-term visual memory across long episodes.

        Each non-first call appends (raw_last_frame, vae_latent) to the buffer, which is trimmed to roughly
        `local_attn_size / num_frame_per_block` entries. Use `warmup_compile()` once after loading to
        pre-compile every distinct KV-cache shape this path exercises.
        """
        start_time = time.perf_counter()

        self.set_frozen_modules_to_eval_mode()
        state_features = data["state"].to(dtype=torch.bfloat16) if self.use_state else None

        videos = self._prepare_video(data["images"])

        _needs_replay = False
        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
            self._history_buffer = []
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
            self._history_buffer = []
        elif self.current_start_frame >= self.dit.local_attn_size:
            _needs_replay = True
            print(f"current_start_frame ({self.current_start_frame}) >= local_attn_size ({self.dit.local_attn_size}), will replay KV cache")

        _started_from_empty_stream = self.current_start_frame == 0

        prompt_emb = self.encode_prompt(data["text"], data["text_attention_mask"])

        # A (re)started stream is anchored on the newest observed frame: the window ends at the
        # current time, which is where the predicted action chunk starts (frame 0 in training).
        image = videos[:, :, -1:].transpose(1, 2)

        _raw_last_frame = image.clone()

        if self.current_start_frame == 0:
            clip_feas, image = self.encode_image(image)
            self.clip_feas = clip_feas.to(dtype=image.dtype)

        assert self.clip_feas is not None, "the stream has no first-frame conditioning"

        if self.current_start_frame != 0:
            # VAE-encode the 8-frame window observed since the last call. The VAE needs 1 + 4k
            # frames, so the first frame is repeated as the causal anchor.
            videos = torch.cat([videos[:, :, 0:1], videos], dim=2)

            with self._vae_gather_ctx():
                image = self.vae.encode(videos)

        noise_obs = self.generate_noise(
            (image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]),
            seed=self.seed, device='cuda', dtype=torch.bfloat16,
        )
        noise_action = self.generate_noise(
            (image.shape[0], self.action_horizon, self.dit.action_dim),
            seed=self.seed, device='cuda', dtype=torch.bfloat16,
        )
        batch_size, num_channels, num_latent_frames, height, width = noise_obs.shape

        frame_seqlen = (height // 2) * (width // 2)
        seq_len = num_latent_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)
        _current_vae_latent = image.clone() if self.current_start_frame != 0 else None

        if self.current_start_frame == 0 or _needs_replay:
            self.kv_cache = self._create_kv_cache(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache is not None
        kv_cache = self.kv_cache

        if _needs_replay:
            buf = self._history_buffer
            print(f"Replay: rebuilding KV cache from {len(buf)} buffered entries")

            # Re-derive clip_feas from the oldest buffered anchor frame so
            # the rebuilt KV cache and the image-encoder conditioning agree.
            clip_feas, cond_latent = self.encode_image(buf[0][0])
            self.clip_feas = clip_feas.to(dtype=cond_latent.dtype)

            self.current_start_frame = 0
            _ts_cond = torch.zeros([batch_size, 1], device=noise_obs.device, dtype=torch.int64)
            self._run_diffusion_steps(
                noisy_input=cond_latent,
                timestep=_ts_cond,
                action=None,
                timestep_action=None,
                context=prompt_emb,
                seq_len=frame_seqlen,
                clip_feature=self.clip_feas,
                kv_cache=kv_cache,
                kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
            )
            self.current_start_frame = 1

            _ts_obs = torch.zeros(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device, dtype=torch.int64,
            )
            for _buf_idx in range(1, len(buf)):
                self.current_start_frame += self.num_frame_per_block
                _sf = self.current_start_frame - self.num_frame_per_block
                _replay_lat = buf[_buf_idx][1][:, -self.num_frame_per_block:]
                self._run_diffusion_steps(
                    noisy_input=_replay_lat.transpose(1, 2),
                    timestep=_ts_obs,
                    action=None,
                    timestep_action=None,
                    context=prompt_emb,
                    seq_len=seq_len,
                    clip_feature=self.clip_feas,
                    kv_cache=kv_cache,
                    kv_cache_metadata=dict(start_frame=_sf, update_kv_cache=True),
                )

            self.current_start_frame += self.num_frame_per_block
            print(f"Replay: done, csf={self.current_start_frame}")

        elif self.current_start_frame == 0:
            timestep = torch.zeros([batch_size, 1], device=noise_obs.device, dtype=torch.int64)
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep,
                action=None,
                timestep_action=None,
                context=prompt_emb,
                seq_len=frame_seqlen,
                clip_feature=self.clip_feas,
                kv_cache=kv_cache,
                kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
            )
            self.current_start_frame += 1

        if self.current_start_frame != 1:
            timestep = torch.zeros([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64)
            current_ref_latents = image[:, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep,
                action=None,
                timestep_action=None,
                context=prompt_emb,
                seq_len=seq_len,
                clip_feature=self.clip_feas,
                kv_cache=kv_cache,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        latents, latents_action, dit_compute_steps = self._denoise(
            noise_obs, noise_action, prompt_emb, state_features, seq_len, kv_cache,
        )

        output = latents
        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        torch.cuda.synchronize()
        print(f"[replay] Total {time.perf_counter() - start_time:.2f}s ({dit_compute_steps} DiT steps)")

        if _current_vae_latent is not None:
            self._history_buffer.append((_raw_last_frame, _current_vae_latent))
        elif _started_from_empty_stream:
            # Single-block checkpoints can hit local_attn_size immediately after
            # the first generated chunk. Keep the initial conditioning frame so
            # the next request can rebuild KV cache via replay.
            self._history_buffer.append((_raw_last_frame, None))
        _max_buf = max(
            1,
            (self.dit.local_attn_size - self.num_frame_per_block - 1)
            // self.num_frame_per_block,
        )
        if len(self._history_buffer) > _max_buf:
            self._history_buffer = self._history_buffer[-_max_buf:]

        return {"action_pred": latents_action, "video_pred": output.transpose(1, 2)}

    @torch.inference_mode()
    def predict_chunk(self, inputs: dict) -> dict:
        """Predict the next action chunk (and video block) for one streaming step.

        Returns `action_pred` (B, action_horizon, action_dim) normalised actions and `video_pred` latents."""
        data = self._prepare_inputs(inputs)
        outputs = self._stream_step_replay(data) if self.replay_kv_cache else self._stream_step(data)
        action = outputs["action_pred"]
        if action.shape[1] != self.action_horizon or action.shape[2] != self.action_dim:
            raise ValueError(f"unexpected action prediction shape {tuple(action.shape)}")
        return outputs

    def reset_stream(self) -> None:
        """Forget the current episode: the next inference call re-encodes the conditioning frame and
        rebuilds the KV caches."""
        self.current_start_frame = 0
        self.language = None
        self.clip_feas = None
        self.kv_cache = None
        self._history_buffer = []

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.dit.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)

        # Torch compile the heavy modules. Each distinct KV-cache length (initial 1-frame prefill,
        # streaming blocks, post-replay shape) gets its own graph, so servers call `warmup_compile()`
        # once after load so no real request pays the compile cost.
        print("Torch compiling the Wan _forward_blocks, TextEncoder, ImageEncoder, and VAE modules.")

        self.dit._forward_blocks = torch.compile(
            mode="reduce-overhead", fullgraph=True, dynamic=False,
        )(self.dit._forward_blocks)

        self.text_encoder.forward = torch.compile(
            mode="reduce-overhead", fullgraph=True, dynamic=False,
        )(self.text_encoder.forward)

        self.image_encoder.model.visual.forward = torch.compile(
            mode="reduce-overhead", fullgraph=True, dynamic=False,
        )(self.image_encoder.model.visual.forward)

        self.vae.model.encode = torch.compile(
            mode="reduce-overhead", fullgraph=True, dynamic=False,
        )(self.vae.model.encode)

    @torch.inference_mode()
    def warmup_compile(self, height: int = 176, width: int = 320) -> None:
        """Run a sequence of dummy `_stream_step_replay` calls so
        that every distinct KV-cache shape (initial 1-frame, mid-buffer 4-frame
        blocks, and the post-trim shape that appears once `current_start_frame`
        exceeds `local_attn_size`) is JIT-compiled BEFORE serving real
        requests. Without this, the first real call after start-up pays the
        full compile cost (tens of seconds) and the second real call hits
        another recompile when the cache transitions to its replay shape.
        All buffers and counters are reset to a clean state on return so the
        first real session starts cold.
        """
        local_attn_size = self.dit.local_attn_size
        num_fpb = self.num_frame_per_block

        # Calls to exercise: 1 initial call + enough 8-frame window calls
        # to fill local_attn_size + 1 sliding-window replay call.
        num_chunks_to_fill = max((local_attn_size - 1 + num_fpb - 1) // num_fpb, 1)
        total_calls = 1 + num_chunks_to_fill + 1

        print(f"[Warmup] Starting torch.compile warmup "
              f"(local_attn_size={local_attn_size}, "
              f"num_frame_per_block={num_fpb}, "
              f"total warmup calls={total_calls})...")

        device = self.device
        dtype = torch.bfloat16
        B = 1
        text_seq_len = 512

        dummy_text = torch.ones(B, text_seq_len, device=device, dtype=torch.long)
        dummy_text_mask = torch.ones(B, text_seq_len, device=device, dtype=torch.long)

        # When use_state is on, the replay path reads data["state"]. The real infer path feeds one state token per
        # register block (state_features.shape[1] == num_blocks *
        # num_state_per_block, with num_blocks = action_horizon //
        # num_action_per_block). Mirror that shape here so the compiled
        # state_encoder / state-attention paths are the ones the production
        # path will hit. Values are zero (warmup output is meaningless).
        dummy_state = None
        if self.use_state:
            num_blocks = max(self.action_horizon // self.dit.num_action_per_block, 1)
            n_state_tokens = num_blocks * self.dit.num_state_per_block
            dummy_state = torch.zeros(
                B, n_state_tokens, self.dit.max_state_dim,
                device=device, dtype=dtype,
            )

        # Start from a clean state so the first call hits the fresh-session branch.
        self.reset_stream()

        warmup_start = time.perf_counter()
        for i in range(total_calls):
            # The servers send an 8-frame window on every call.
            num_video_frames = 8

            dummy_images = torch.randint(
                0, 256,
                (B, num_video_frames, height, width, 3),
                device=device, dtype=torch.uint8,
            )

            data = {
                "images": dummy_images,
                "text": dummy_text,
                "text_attention_mask": dummy_text_mask,
            }
            if dummy_state is not None:
                data["state"] = dummy_state

            call_start = time.perf_counter()
            print(f"[Warmup] Call {i + 1}/{total_calls} "
                  f"(frames={num_video_frames}, csf={self.current_start_frame})...")

            self._stream_step_replay(data)
            call_time = time.perf_counter() - call_start
            print(f"[Warmup] Call {i + 1}/{total_calls} done in {call_time:.1f}s")

        total_time = time.perf_counter() - warmup_start

        # Final clean state so the next real session starts cold.
        self.reset_stream()

        print(f"[Warmup] torch.compile warmup complete: {total_calls} calls "
              f"compiled in {total_time:.1f}s. Inference is now at full speed.")
