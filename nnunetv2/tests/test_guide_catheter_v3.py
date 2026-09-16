import unittest
from pathlib import Path
from types import SimpleNamespace

import nnunetv2
import numpy as np
import torch
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerGuideCatheterV3 import (
    _initial_patch_size_2d,
    nnUNetTrainerGuideCatheterV3,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def training_transforms(trainer):
    return trainer.get_training_transforms(
        patch_size=(32, 32),
        rotation_for_DA=(-np.pi, np.pi),
        deep_supervision_scales=None,
        mirror_axes=(0, 1),
        do_dummy_2d_data_aug=False,
    )


class TestGuideCatheterV3(unittest.TestCase):
    def test_cli_discovery(self):
        trainer = recursive_find_python_class(
            str(Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"),
            "nnUNetTrainerGuideCatheterV3",
            "nnunetv2.training.nnUNetTrainer",
        )
        self.assertIs(trainer, nnUNetTrainerGuideCatheterV3)

    def test_pipeline_and_sampling(self):
        original = training_transforms(nnUNetTrainer)
        custom = training_transforms(nnUNetTrainerGuideCatheterV3)
        self.assertEqual(
            [type(t) for t in original.transforms], [type(t) for t in custom.transforms]
        )
        for before, after in zip(original.transforms, custom.transforms):
            if isinstance(after, SpatialTransform):
                expected = vars(before).copy()
                expected.update(p_scaling=1.0, scaling=(0.5, 2.0),
                                p_synchronize_scaling_across_axes=1.0)
                self.assertEqual(vars(after), expected)
            else:
                self.assertEqual(repr(before), repr(after))

        spatial = next(t for t in custom.transforms if isinstance(t, SpatialTransform))
        random_state = np.random.get_state()
        try:
            np.random.seed(9)
            scales = []
            for _ in range(128):
                affine = spatial.get_parameters(image=torch.ones(1, 96, 96))["affine"]
                self.assertIsNotNone(affine)
                singular_values = np.linalg.svd(affine, compute_uv=False)
                np.testing.assert_allclose(singular_values[0], singular_values[1])
                scales.append(singular_values[0])
            self.assertTrue(all(0.5 <= scale <= 2.0 for scale in scales))
            self.assertTrue(all(scale != 1.0 for scale in scales))
            self.assertLess(min(scales), 0.6)
            self.assertGreater(max(scales), 1.9)
        finally:
            np.random.set_state(random_state)

    def test_crop_covers_rotations_and_scales(self):
        for shape, limit in [((32, 32), np.pi), ((32, 40), np.pi),
                             ((32, 80), np.pi / 12)]:
            initial = _initial_patch_size_2d(shape, (-limit, limit), 2.0)
            angles = list(np.linspace(-limit, limit, 13))
            angles += [min(limit, np.arctan2(*shape)),
                       min(limit, np.arctan2(*shape[::-1]))]
            for scale in (0.5, 1.0, 2.0):
                for angle in angles:
                    with self.subTest(shape=shape, scale=scale, angle=angle):
                        spatial = SpatialTransform(
                            shape, 0, False, p_rotation=1, rotation=(angle, angle),
                            p_scaling=1, scaling=(scale, scale),
                            p_synchronize_scaling_across_axes=1,
                            bg_style_seg_sampling=False,
                            border_mode_seg="constant", padding_value_seg=-1,
                        )
                        result = spatial(
                            image=torch.ones(1, *initial),
                            segmentation=torch.ones(1, *initial, dtype=torch.int16),
                        )
                        self.assertEqual(tuple(result["image"].shape), (1, *shape))
                        torch.testing.assert_close(result["image"], torch.ones(1, *shape))
                        self.assertTrue(torch.all(result["segmentation"] == 1))

    def test_image_and_label_stay_aligned(self):
        shape = (32, 32)
        initial = _initial_patch_size_2d(shape, (-np.pi, np.pi), 2.0)
        label = torch.zeros(1, *initial, dtype=torch.int16)
        cy, cx = (int(n // 2) for n in initial)
        label[:, cy - 10:cy + 10, cx - 6:cx + 6] = 1
        for scale in (0.5, 2.0):
            spatial = SpatialTransform(
                shape, 0, False, p_rotation=1, rotation=(0.4, 0.4),
                p_scaling=1, scaling=(scale, scale),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg="constant", padding_value_seg=-1,
            )
            result = spatial(image=label.float(), segmentation=label.clone())
            self.assertTrue(torch.equal(result["image"] >= 0.5,
                                        result["segmentation"].bool()))
            self.assertEqual(set(result["segmentation"].unique().tolist()), {0, 1})

    def test_trainer_uses_geometry_and_rejects_3d(self):
        trainer = object.__new__(nnUNetTrainerGuideCatheterV3)
        trainer.configuration_manager = SimpleNamespace(patch_size=[512, 512])
        trainer.print_to_log_file = lambda *args: None
        rotation, dummy_2d, initial, mirrors = (
            trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        np.testing.assert_array_equal(initial, [1451, 1451])
        self.assertEqual(rotation, (-np.pi, np.pi))
        self.assertFalse(dummy_2d)
        self.assertEqual(mirrors, (0, 1))
        trainer.configuration_manager.patch_size = [16, 32, 32]
        with self.assertRaisesRegex(ValueError, "2D configuration"):
            trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        with self.assertRaisesRegex(ValueError, "2D configuration"):
            nnUNetTrainerGuideCatheterV3.get_training_transforms((16, 32, 32))


if __name__ == "__main__":
    unittest.main()
