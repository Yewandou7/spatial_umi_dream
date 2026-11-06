import unittest

import numpy as np

from spatial_umi_dream.data.dataset.sharded import (
    CHUNK_SIZE,
    VIDEO_FRAME_OFFSETS,
    _last_full_forward_anchor,
    _plan_forward_chunk_anchors,
)


def _expand_plan(anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video = (anchors[:, None] + VIDEO_FRAME_OFFSETS[None, :]).reshape(-1)
    video = np.append(video, anchors[-1] + CHUNK_SIZE)
    action = (
        anchors[:, None] + np.arange(CHUNK_SIZE, dtype=np.int64)[None, :]
    ).reshape(-1)
    return video, anchors, action


class ForwardChunkPlannerTest(unittest.TestCase):
    def test_aligned_fixed_shapes(self) -> None:
        language = np.full(107, "same")
        anchors = _plan_forward_chunk_anchors(10, 107, 4, language)
        video, state, action = _expand_plan(anchors)

        np.testing.assert_array_equal(anchors, [10, 34, 58, 82])
        self.assertEqual(video.shape, (33,))
        self.assertEqual(state.shape, (4,))
        self.assertEqual(action.shape, (96,))
        self.assertEqual(video[-1], 106)
        np.testing.assert_array_equal(state, video[:-1:8])
        np.testing.assert_array_equal(state, action[::CHUNK_SIZE])

    def test_final_video_frame_boundary(self) -> None:
        self.assertEqual(_last_full_forward_anchor(97, 4), 0)
        np.testing.assert_array_equal(
            _plan_forward_chunk_anchors(0, 97, 4), [0, 24, 48, 72]
        )
        self.assertEqual(_plan_forward_chunk_anchors(0, 96, 4).size, 0)
        self.assertEqual(_plan_forward_chunk_anchors(1, 97, 4).size, 0)

    def test_future_language_change(self) -> None:
        language = np.full(121, "first")
        language[24:] = "second"
        self.assertEqual(
            _plan_forward_chunk_anchors(0, 121, 4, language).size, 0
        )

    def test_invalid_requests(self) -> None:
        self.assertEqual(_plan_forward_chunk_anchors(-1, 100, 4).size, 0)
        self.assertEqual(_plan_forward_chunk_anchors(100, 100, 4).size, 0)
        with self.assertRaisesRegex(ValueError, "max_chunks must be positive"):
            _plan_forward_chunk_anchors(0, 100, 0)


if __name__ == "__main__":
    unittest.main()
