"""CPU regressions for state visibility in teacher forcing and cached inference.

Use the production projections, RoPE, and attention routing with PyTorch SDPA.
No model checkpoints or GPU kernels are needed.
"""

import os
import unittest
from unittest.mock import patch

import torch

from spatial_umi_dream.model.wan import dit
from spatial_umi_dream.model.wan.attention import AttentionModule


class StateAttentionConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"ATTENTION_BACKEND": "torch"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(123)
        self.freqs = dit.rope_params(128, 4)

    def make_model(self, use_state=True):
        model = dit.CausalWanSelfAttention(
            dim=8, num_heads=2, frame_seqlen=2,
            num_frame_per_block=2, num_action_per_block=24,
            num_state_per_block=int(use_state), qk_norm=True,
        ).eval()
        # Keep the real backend but avoid bfloat16 rounding in parity checks.
        model.attn = AttentionModule(dtype=torch.float32)
        model.causal_attn = AttentionModule(dtype=torch.float32, causal=True)
        return model

    def cached_forward(self, model, clean, video, action, state, block_index):
        start_frame = 1 + 2 * block_index
        start_token = start_frame * model.frame_seqlen
        video_freqs = self.freqs[:len(clean[0])].unsqueeze(1)
        empty_cache = torch.empty(2, 1, 0, 2, 4)
        # Build actual video K/V using the same projections and RoPE as prefill.
        _, cache = model(
            clean[:, :start_token], freqs=video_freqs[:start_token],
            freqs_action=self.freqs, action_register_length=None,
            kv_cache=empty_cache, current_start_frame=0,
        )
        tokens = [video, action] + ([state] if state is not None else [])
        return model(
            torch.cat(tokens, dim=1),
            freqs=video_freqs[start_token:start_token + video.shape[1]],
            freqs_action=self.freqs,
            freqs_state=self.freqs if state is not None else None,
            action_register_length=action.shape[1] + (state.shape[1] if state is not None else 0),
            kv_cache=cache, current_start_frame=start_frame,
        )

    def test_teacher_forcing_matches_cached_inference(self):
        # Two blocks exercise both initial and later action/state RoPE indices.
        for use_state in (False, True):
            with self.subTest(use_state=use_state):
                model = self.make_model(use_state)
                clean = torch.randn(1, 10, 8)  # five frames, two tokens per frame
                noisy = torch.randn_like(clean)
                actions = torch.randn(1, 48, 8)
                states = torch.randn(1, 2, 8) if use_state else None
                tokens = [clean, noisy, actions] + ([states] if use_state else [])
                trained, _ = model(
                    torch.cat(tokens, dim=1), freqs=self.freqs[:10].unsqueeze(1),
                    freqs_action=self.freqs, freqs_state=self.freqs if use_state else None,
                    action_register_length=48 + (2 if use_state else 0),
                    kv_cache=None,
                )
                for block in range(2):
                    with self.subTest(block=block):
                        start = 2 + 4 * block
                        action_start = 24 * block
                        inferred, _ = self.cached_forward(
                            model, clean, noisy[:, start:start + 4],
                            actions[:, action_start:action_start + 24],
                            states[:, block:block + 1] if use_state else None, block,
                        )
                        torch.testing.assert_close(
                            inferred[:, :4], trained[:, 10 + start:10 + start + 4],
                            rtol=1e-5, atol=1e-6,
                        )
                        torch.testing.assert_close(
                            inferred[:, 4:28], trained[:, 20 + action_start:20 + action_start + 24],
                            rtol=1e-5, atol=1e-6,
                        )
                        if use_state:
                            torch.testing.assert_close(
                                inferred[:, -1:], trained[:, 68 + block:69 + block],
                                rtol=1e-5, atol=1e-6,
                            )

    def test_state_conditions_predictions_without_entering_video_cache(self):
        model = self.make_model()
        clean, video, action = (torch.randn(1, n, 8) for n in (10, 4, 24))
        state = torch.randn(1, 1, 8, requires_grad=True)
        for block in range(2):
            with self.subTest(block=block):
                original, cache = self.cached_forward(model, clean, video, action, state, block)
                changed, other_cache = self.cached_forward(model, clean, video, action, -state, block)
                self.assertGreater((original[:, :4] - changed[:, :4]).abs().max().item(), 1e-6)
                self.assertGreater((original[:, 4:28] - changed[:, 4:28]).abs().max().item(), 1e-6)
                gradient = torch.autograd.grad(original[:, 4:28].square().sum(), state)[0]
                self.assertGreater(gradient.abs().max().item(), 1e-6)
                torch.testing.assert_close(cache, other_cache, rtol=0, atol=0)
                self.assertEqual(cache.shape[2], (1 + 2 * block) * 2 + 4)

    def test_state_queries_remain_self_only(self):
        model = self.make_model()
        clean, video, action, state = (torch.randn(1, n, 8) for n in (10, 4, 24, 1))
        original, _ = self.cached_forward(model, clean, video, action, state, 0)
        changed, _ = self.cached_forward(model, -clean, -video, -action, state, 0)
        torch.testing.assert_close(original[:, -1:], changed[:, -1:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
