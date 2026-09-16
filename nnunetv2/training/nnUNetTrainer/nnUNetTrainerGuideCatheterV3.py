import numpy as np
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _initial_patch_size_2d(patch_size, rotation, max_scale):
    """Bound the source footprint over the symmetric rotation interval."""
    shape = np.asarray(patch_size, dtype=float)
    max_angle = min(max(abs(angle) for angle in rotation), np.pi / 2)
    # Each axis reaches its largest extent at atan(other_axis / this_axis).
    angles = np.minimum(max_angle, np.arctan2(shape[::-1], shape))
    extent = shape * np.cos(angles) + shape[::-1] * np.sin(angles)
    # SpatialTransform scales sampling coordinates: larger scales zoom out.
    return np.ceil(max_scale * extent).astype(int) + 2


class nnUNetTrainerGuideCatheterV3(nnUNetTrainer):
    """2D guide catheter training with isotropic scaling on every patch."""

    scale_range = (0.5, 2.0)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 2:
            raise ValueError("nnUNetTrainerGuideCatheterV3 requires a 2D configuration")
        rotation, dummy_2d, _, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        initial_patch_size = _initial_patch_size_2d(
            patch_size, rotation, max(self.scale_range)
        )
        self.print_to_log_file(
            "\n========== GUIDE CATHETER V3 CUSTOM TRAINER ==========\n"
            f"Trainer: {self.__class__.__name__}\n"
            f"Isotropic scaling: {self.scale_range[0]}-"
            f"{self.scale_range[1]} on 100% of training patches (uniform sampling)\n"
            f"Loader crop: {initial_patch_size.tolist()}; network patch: {patch_size}\n"
            "====================================================\n"
        )
        return rotation, dummy_2d, initial_patch_size, mirror_axes

    @classmethod
    def get_training_transforms(cls, patch_size, *args, **kwargs):
        if len(patch_size) != 2:
            raise ValueError("nnUNetTrainerGuideCatheterV3 requires a 2D configuration")
        transforms = super().get_training_transforms(patch_size, *args, **kwargs)
        spatial = [t for t in transforms.transforms if isinstance(t, SpatialTransform)]
        if len(spatial) != 1:
            raise RuntimeError("Expected exactly one SpatialTransform in the training pipeline")
        spatial[0].p_scaling = 1.0
        spatial[0].scaling = cls.scale_range
        spatial[0].p_synchronize_scaling_across_axes = 1.0
        return transforms
