"""Numpy SE(3) helpers for rot6d-format pose deltas in the data pipeline.

The data loader receives action / state already in rot6d format
(`[xyz(3), rot6d(6)]`) because the upstream LeRobot conversion replaces the
quaternion column with rot6d offline. These helpers compute per-chunk
SE(3)-relative deltas while staying in rot6d the whole way.

No quaternion functions and no torch on purpose: quat math is in the offline
conversion script, and the data pipeline runs in numpy in DataLoader workers.
"""

import numpy as np


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """6D representation to 3x3 rotation matrix via Gram-Schmidt.

    Input shape `(..., 6)`, output `(..., 3, 3)`. Columns of the returned
    matrix are `[b1, b2, b3]` where `b1` is the normalized first three
    input dims, `b2` is the second three with the `b1` component projected
    out then normalized, and `b3 = b1 x b2`.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to 6D (first two columns flattened).

    Input shape `(..., 3, 3)`, output `(..., 6)`.
    """
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def pose9_to_mat(pose9: np.ndarray) -> np.ndarray:
    """`[xyz(3), rot6d(6)]` to 4x4 homogeneous SE(3) matrix.

    Input shape `(..., 9)`, output `(..., 4, 4)`.
    """
    pose9 = np.asarray(pose9, dtype=np.float64)
    shape = pose9.shape[:-1]
    T = np.zeros(shape + (4, 4), dtype=np.float64)
    T[..., :3, :3] = rot6d_to_rotmat(pose9[..., 3:9])
    T[..., :3, 3] = pose9[..., :3]
    T[..., 3, 3] = 1.0
    return T


def relative_pose9(chunk_pose9: np.ndarray, base_pose9: np.ndarray) -> np.ndarray:
    """Per-chunk SE(3) relative pose, output stays in rot6d format.

    Computes `T_rel = inv(T_base) @ T_chunk` per timestep then re-encodes
    rotation back to rot6d.

    Args:
        chunk_pose9: `(H, 9)` absolute poses `[xyz, rot6d]` per timestep.
        base_pose9: `(9,)` anchor pose for the chunk (typically the first
            timestep's state).

    Returns:
        `(H, 9)` relative poses `[xyz_rel, rot6d_rel]`. The dtype matches
        the input; intermediate math is float64.
    """
    chunk_pose9 = np.asarray(chunk_pose9)
    base_pose9 = np.asarray(base_pose9)
    T_base = pose9_to_mat(base_pose9)
    T_base_inv = np.linalg.inv(T_base)
    T_chunk = pose9_to_mat(chunk_pose9)
    T_rel = np.matmul(T_base_inv, T_chunk)
    pos = T_rel[..., :3, 3]
    rot6d = rotmat_to_rot6d(T_rel[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1).astype(chunk_pose9.dtype, copy=False)


def absolute_pose9(chunk_rel9: np.ndarray, base_pose9: np.ndarray) -> np.ndarray:
    """Inverse of `relative_pose9`. Per-chunk SE(3) absolute pose from a
    rot6d-format relative pose composed with an anchor.

    Computes `T_abs = T_base @ T_rel` per timestep then re-encodes rotation
    back to rot6d. Use at inference time to convert model-predicted relative
    actions back to absolute poses given the robot's current state.

    Args:
        chunk_rel9: `(H, 9)` relative poses `[xyz_rel, rot6d_rel]` per
            timestep (typically the model's action output).
        base_pose9: `(9,)` anchor pose `[xyz, rot6d]` (typically the
            robot's current proprioceptive state when issuing the chunk).

    Returns:
        `(H, 9)` absolute poses `[xyz_abs, rot6d_abs]`. The dtype matches
        the input; intermediate math is float64.
    """
    chunk_rel9 = np.asarray(chunk_rel9)
    base_pose9 = np.asarray(base_pose9)
    T_base = pose9_to_mat(base_pose9)
    T_rel = pose9_to_mat(chunk_rel9)
    T_abs = np.matmul(T_base, T_rel)
    pos = T_abs[..., :3, 3]
    rot6d = rotmat_to_rot6d(T_abs[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1).astype(chunk_rel9.dtype, copy=False)
