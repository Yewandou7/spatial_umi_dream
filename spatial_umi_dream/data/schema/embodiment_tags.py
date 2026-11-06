from enum import Enum


class EmbodimentTag(Enum):
    """Embodiments the release ships data configs and checkpoints for.

    The tag value keys the per-embodiment modality configs / transforms in the data
    configs and the `experiment_cfg/metadata.json` saved next to each checkpoint.
    """

    UMI = "umi"
    """
    The UMI bimanual handheld dataset (rot6d-format pose + gripper per arm).
    """

    ROBOTWIN = "robotwin"
    """
    The RobotWin bimanual dataset (rot6d-format pose + gripper per arm, same layout as UMI).
    """
