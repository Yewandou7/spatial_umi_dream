from pathlib import Path
from hydra import compose, initialize_config_dir
import numpy as np
import pytest

from spatial_umi_dream.data.canvas import (
    RGB_VIDEO_KEYS,
    VIDEO_KEYS,
    assemble_rgbd_canvas,
    robotwin_prompt_with_layout,
    umi_prompt_with_layout,
    validate_camera_frame,
    validate_video_keys,
)


CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "spatial_umi_dream" / "configs").resolve()
)


def _solid_video(values: list[int], *, time: int = 2, height: int = 3, width: int = 5):
    views = [
        np.full((time, height, width, 3), value, dtype=np.uint8)
        for value in values
    ]
    return np.stack(views, axis=1)


def _compose(extra_overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="train",
            overrides=["data=umi", "umi_data_roots=[/dataset/umi]", *(extra_overrides or [])],
        )


def test_four_streams_land_in_eye_paired_quadrants():
    video = _solid_video([11, 22, 33, 44])
    canvas = assemble_rgbd_canvas(video, target_height=6, target_width=10)

    assert canvas.shape == (2, 6, 10, 3)
    np.testing.assert_array_equal(canvas[:, :3, :5], 11)
    np.testing.assert_array_equal(canvas[:, :3, 5:], 33)
    np.testing.assert_array_equal(canvas[:, 3:, :5], 22)
    np.testing.assert_array_equal(canvas[:, 3:, 5:], 44)

    # PolicyInputTransform adds the singleton view and moves channels before H/W.
    collator_input = canvas.transpose(0, 3, 1, 2)[None]
    assert collator_input.shape == (1, 2, 3, 6, 10)


@pytest.mark.parametrize("view_count", [0, 1, 2, 3, 5])
def test_canvas_rejects_missing_or_extra_views(view_count):
    video = np.zeros((1, view_count, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expects exactly 4 views"):
        assemble_rgbd_canvas(video)


def test_canvas_rejects_wrong_channels_and_geometry():
    bad_channels = np.zeros((1, 4, 2, 2, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="three-channel"):
        assemble_rgbd_canvas(bad_channels)

    video = _solid_video([1, 2, 3, 4], time=1, height=2, width=2)
    with pytest.raises(ValueError, match="configured Wan input geometry"):
        assemble_rgbd_canvas(video, target_height=8, target_width=8)


def test_key_contract_supports_dataset_specific_names_but_checks_structure():
    assert validate_video_keys(VIDEO_KEYS) == VIDEO_KEYS

    custom = ("video.left_rgb", "video.right_rgb", "video.left_d", "video.right_d")
    assert validate_video_keys(custom) == custom
    with pytest.raises(ValueError, match="exactly 4 ordered video keys"):
        validate_video_keys(custom[:3])
    with pytest.raises(ValueError, match="exactly 4 ordered video keys"):
        validate_video_keys(RGB_VIDEO_KEYS)
    with pytest.raises(ValueError, match="must be unique"):
        validate_video_keys((custom[0], custom[1], custom[2], custom[2]))


def test_umi_prompt_describes_the_actual_quadrants():
    prompt = umi_prompt_with_layout("Pick up the cup")
    assert prompt.startswith("A four-panel RGB-and-depth video")
    assert "top-left panel is the left wrist RGB view" in prompt
    assert "top-right panel is its depth visualization" in prompt
    assert "bottom-left panel is the right wrist RGB view" in prompt
    assert "bottom-right panel is its depth visualization" in prompt
    assert umi_prompt_with_layout(prompt) == prompt


def test_robotwin_prompt_describes_the_actual_quadrants():
    prompt = robotwin_prompt_with_layout("['Shake the bottle']")
    assert prompt.startswith("A RobotWin four-panel RGB-and-depth video shows a bimanual robot that must shake the bottle.")
    assert "top-right panel is its depth visualization" in prompt
    assert robotwin_prompt_with_layout(prompt) == prompt


def test_deployment_camera_frames_require_raw_uint8():
    key = "video.robot0_depth"
    frame = np.zeros((3, 5, 3), dtype=np.uint8)
    assert validate_camera_frame(frame, camera_key=key) is frame

    normalized_depth = np.full((3, 5, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="Refusing to cast non-uint8 camera data"):
        validate_camera_frame(normalized_depth, camera_key=key)
    with pytest.raises(ValueError, match=r"must be \(H, W, 3\) uint8"):
        validate_camera_frame(np.zeros((3, 5), dtype=np.uint8), camera_key=key)


def _transforms_by_name(cfg):
    return {
        item._target_.rsplit(".", 1)[-1]: item
        for item in cfg.transforms.umi.transforms
        if "_target_" in item
    }


def test_hydra_config_splits_geometry_from_photometric():
    spatial = _compose()
    all_keys = list(VIDEO_KEYS)
    rgb_keys = list(RGB_VIDEO_KEYS)
    assert list(spatial.modality_configs.umi.video.modality_keys) == all_keys
    assert list(spatial.rgb_video_keys) == all_keys[:2]
    assert spatial.image_resolution_height == 160
    assert spatial.image_resolution_width == 160
    assert spatial.policy_transform.canvas_height == 320
    assert spatial.policy_transform.canvas_width == 320
    assert spatial.model.config.target_video_height == 320
    assert spatial.model.config.target_video_width == 320
    assert spatial.relative_action is True
    assert list(spatial.relative_action_keys) == ["left_pose", "right_pose"]
    assert spatial.relative_state is False
    assert spatial.max_chunk_size == 4
    assert spatial.train_dataset.dataset_class.endswith("ShardedLeRobotSE3DeltaSingleDataset")
    assert spatial.train_dataset.mixture_kwargs.seed == 42

    by_name = _transforms_by_name(spatial)
    assert list(by_name["VideoCrop"].apply_to) == all_keys
    assert list(by_name["VideoResize"].apply_to) == all_keys
    assert list(by_name["VideoColorJitter"].apply_to) == rgb_keys
    assert list(by_name["VideoToTensor"].apply_to) == all_keys
    assert list(by_name["VideoToNumpy"].apply_to) == all_keys
    assert list(by_name["ConcatTransform"].video_concat_order) == all_keys


def test_multi_dataset_roots_use_one_step_proportional_union():
    roots = ["/dataset/filtered_p995", "/dataset/dual_depth"]
    multi = _compose([f"umi_data_roots=[{','.join(roots)}]"])

    mixture = multi.train_dataset.mixture_spec
    assert len(mixture) == 1
    assert list(mixture[0].dataset_path.umi) == roots
    assert mixture[0].dataset_weight == 1.0
    assert mixture[0].distribute_weights is True
    assert multi.train_dataset.mixture_kwargs.balance_dataset_weights is False


def test_multi_dataset_override_accepts_explicit_sampling_weights():
    weighted_mixture = (
        "[{dataset_path:{umi:[/dataset/task_a]},dataset_weight:0.7,"
        "distribute_weights:false},{dataset_path:{umi:[/dataset/task_b]},"
        "dataset_weight:0.3,distribute_weights:false}]"
    )
    multi = _compose([f"train_dataset.mixture_spec={weighted_mixture}"])

    mixture = multi.train_dataset.mixture_spec
    assert len(mixture) == 2
    assert list(mixture[0].dataset_path.umi) == ["/dataset/task_a"]
    assert list(mixture[1].dataset_path.umi) == ["/dataset/task_b"]
    assert [entry.dataset_weight for entry in mixture] == [0.7, 0.3]
    assert all(entry.distribute_weights is False for entry in mixture)
    assert multi.train_dataset.mixture_kwargs.balance_dataset_weights is False


def test_robotwin_config_reads_four_streams_into_160px_panels():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["data=robotwin_umi", "robotwin_data_roots=[/dataset/rt]"],
        )
    assert list(cfg.modality_configs.robotwin.video.modality_keys) == list(VIDEO_KEYS)
    assert list(cfg.rgb_video_keys) == list(RGB_VIDEO_KEYS)
    assert cfg.image_resolution_height == 160
    assert cfg.image_resolution_width == 160
    assert cfg.policy_transform.canvas_height == 320
