"""Training entry: `torchrun ... spatial_umi_dream/train/experiment.py <hydra overrides>`.

See scripts/train/train_umi.sh and scripts/train/train_robotwin_umi.sh for the launch recipes.
"""

import logging
import tempfile
from pathlib import Path

import hydra
from torch.distributed.elastic.multiprocessing.errors import record

from spatial_umi_dream.train.base import BaseExperiment, BaseTrainer
from spatial_umi_dream.train.dataloader_media import (
    optimizer_step_for_current_batch,
    prepare_dataloader_wandb_video,
    should_log_dataloader_video,
    write_dataloader_video_mp4,
)

logger = logging.getLogger(__name__)


class WorldActionTrainer(BaseTrainer):
    """BaseTrainer plus optional W&B logging of the collated video batch (rank 0)."""

    def __init__(self, **kwargs):
        self.wandb_dataloader_log_every_steps = int(
            kwargs.pop("wandb_dataloader_log_every_steps", 0)
        )
        self.wandb_dataloader_video_fps = int(
            kwargs.pop("wandb_dataloader_video_fps", 10)
        )
        if self.wandb_dataloader_log_every_steps < 0:
            raise ValueError("wandb_dataloader_log_every_steps must be non-negative")
        if self.wandb_dataloader_video_fps <= 0:
            raise ValueError("wandb_dataloader_video_fps must be positive")
        self._last_wandb_dataloader_step: int | None = None

        super().__init__(**kwargs)

    def _maybe_log_dataloader_video(self, inputs) -> None:
        # During training_step, Trainer.state.global_step is the number of
        # completed optimizer updates. This batch contributes to the next one.
        optimizer_step = optimizer_step_for_current_batch(
            int(self.state.global_step)
        )
        if not should_log_dataloader_video(
            step=optimizer_step,
            every_steps=self.wandb_dataloader_log_every_steps,
            last_logged_step=self._last_wandb_dataloader_step,
        ):
            return
        if not self.is_world_process_zero():
            return

        # Mark the step before encoding so gradient accumulation cannot retry the
        # same optimizer step when media logging is unavailable or fails.
        self._last_wandb_dataloader_step = optimizer_step
        video_path: Path | None = None
        try:
            import wandb

            if wandb.run is None:
                logger.warning(
                    "Skipping dataloader video at optimizer step %s: no active W&B run",
                    optimizer_step,
                )
                return
            video = prepare_dataloader_wandb_video(inputs["images"])
            media_dir = Path(self.args.output_dir) / ".wandb_dataloader_media"
            media_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f"step_{optimizer_step}_",
                suffix=".mp4",
                dir=media_dir,
                delete=False,
            ) as video_file:
                video_path = Path(video_file.name)
            write_dataloader_video_mp4(
                video,
                video_path,
                fps=self.wandb_dataloader_video_fps,
            )
            wandb.log(
                {
                    "dataloader/video": wandb.Video(
                        str(video_path),
                        format="mp4",
                        caption=(
                            f"optimizer_step={optimizer_step}; collated sample 0; "
                            f"TCHW={tuple(video.shape)}"
                        ),
                    ),
                    # Transformers' W&B callback defines train/global_step as
                    # the step metric for every series. Include it explicitly
                    # so this media point lands exactly on the intended update.
                    "train/global_step": optimizer_step,
                    "dataloader/global_step": optimizer_step,
                }
            )
        except Exception:
            logger.exception(
                "Failed to log dataloader video at optimizer step %s",
                optimizer_step,
            )
        finally:
            if video_path is not None:
                video_path.unlink(missing_ok=True)

    def training_step(self, model, inputs, *args, **kwargs):
        loss_dict = super().training_step(model, inputs, *args, **kwargs)
        # Log only after this exact batch has completed forward/backward.
        self._maybe_log_dataloader_video(inputs)
        return loss_dict


@hydra.main(config_path="../configs", config_name="train", version_base=None)
@record
def main(cfg):
    experiment = BaseExperiment(cfg)
    experiment.train()


if __name__ == "__main__":
    main()
