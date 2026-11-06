# Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# and converted to flow matching by the Alibaba Wan team. Trimmed to the configuration used here:
# x0 prediction, "bh2" solver, lower-order final steps, final sigma 0.

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """UniPC multistep sampler for flow matching.

    A predictor (UniP) of order `solver_order` plus a corrector (UniC) that re-uses the previous step's
    model output, giving an effective order of `solver_order + 1`.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        shift: float = 1.0,
    ):
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device='cuda')
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, shift: float | None = None):
        """Build the inference schedule: `num_inference_steps` shifted sigmas followed by a final sigma of 0."""
        self.num_inference_steps = num_inference_steps

        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
        if shift is None:
            shift = self.config.shift
        assert isinstance(shift, float)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma):
        return 1 - sigma, sigma

    def convert_model_output(self, model_output: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        """Flow prediction -> x0 estimate, which is what the UniPC updates operate on."""
        sigma_t = self.sigmas[step_index]
        x0_pred = sample - sigma_t * model_output
        return x0_pred

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        """One UniP (B(h) version) step of the given order: predicts the sample at the next sigma."""
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
            else:
                rhos_p = torch.linalg.solve_ex(R[:-1, :-1], b[:-1])[0].to(x.dtype)
        else:
            D1s = None
            rhos_p = None

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        """One UniC (B(h) version) step: corrects `this_sample` using the model output evaluated at it."""
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
        else:
            rhos_c = torch.linalg.solve_ex(R, b)[0].to(x.dtype)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def step(self, model_output: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        """Advance `sample` from `self.sigmas[step_index]` to `self.sigmas[step_index + 1]` given the model's
        flow prediction at `step_index`. Returns the previous (less noisy) sample."""
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        use_corrector = step_index > 0 and self.last_sample is not None

        model_output_convert = self.convert_model_output(
            model_output=model_output,
            sample=sample,
            step_index=step_index,
        )
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=step_index,
            )
            # We must clone the outputs of a CUDA graph'd computation.
            sample = sample.clone()

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]

        self.model_outputs[-1] = model_output_convert

        # Lower-order solvers in the final steps stabilise sampling with few steps.
        this_order = min(
            self.config.solver_order,
            len(self.timesteps) - step_index,
        )

        # Warmup for multistep.
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        # Pass the original non-converted model output.
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
            step_index=step_index,
        )
        # We must clone the outputs of a CUDA graph'd computation.
        prev_sample = prev_sample.clone()

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        return prev_sample
