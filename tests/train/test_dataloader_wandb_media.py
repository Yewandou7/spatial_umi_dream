import numpy as np
import pytest
import torch

from spatial_umi_dream.train.dataloader_media import (
    optimizer_step_for_current_batch,
    prepare_dataloader_wandb_video,
    should_log_dataloader_video,
    write_dataloader_video_mp4,
)


@pytest.mark.parametrize(
    ("completed_steps", "optimizer_step"),
    [(0, 1), (999, 1000), (1000, 1001)],
)
def test_optimizer_step_for_current_batch(completed_steps, optimizer_step):
    assert optimizer_step_for_current_batch(completed_steps) == optimizer_step


def test_prepare_dataloader_wandb_video_selects_sample_zero_and_returns_tchw():
    images = torch.arange(2 * 3 * 2 * 4 * 3, dtype=torch.uint8).reshape(
        2, 3, 2, 4, 3
    )

    video = prepare_dataloader_wandb_video(images)

    assert video.shape == (3, 3, 2, 4)
    assert video.dtype == np.uint8
    np.testing.assert_array_equal(video[:, 0], images[0, :, :, :, 0].numpy())
    np.testing.assert_array_equal(video[:, 2], images[0, :, :, :, 2].numpy())


@pytest.mark.parametrize(
    ("step", "every_steps", "last_logged_step", "expected"),
    [
        (0, 1000, None, False),
        (999, 1000, None, False),
        (1000, 1000, None, True),
        (1000, 1000, 1000, False),
        (2000, 1000, 1000, True),
        (1000, 0, None, False),
    ],
)
def test_should_log_dataloader_video(
    step, every_steps, last_logged_step, expected
):
    assert (
        should_log_dataloader_video(
            step=step,
            every_steps=every_steps,
            last_logged_step=last_logged_step,
        )
        is expected
    )


def test_prepare_dataloader_wandb_video_rejects_non_uint8():
    with pytest.raises(TypeError, match="uint8"):
        prepare_dataloader_wandb_video(torch.zeros(1, 3, 2, 4, 3))


def test_write_dataloader_video_mp4(tmp_path):
    video = np.zeros((2, 3, 16, 16), dtype=np.uint8)
    video[1, 0] = 255
    output_path = tmp_path / "dataloader.mp4"

    write_dataloader_video_mp4(video, output_path, fps=10)

    assert output_path.stat().st_size > 0
    assert b"ftyp" in output_path.read_bytes()[:32]
