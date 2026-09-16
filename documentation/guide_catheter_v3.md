# Guide catheter V3 trainer

Based on nnU-Net v2.7.0 (566198c4f3f8190b0d52a10172c84a5cd8f78db2).

After staging and preprocessing the corrected training data, select the trainer:

```bash
nnUNetv2_train 001 2d all -tr nnUNetTrainerGuideCatheterV3 --npz
```

`nnUNetTrainerGuideCatheterV3` applies isotropic in-plane scaling to every training
patch. The sampling-coordinate factor is uniform in [0.5, 2.0]: 0.5 doubles object
size, while 2.0 halves it. This is linear uniform sampling, not log-uniform. The
image and segmentation receive the same transform. All other augmentation,
validation, and training settings are inherited from the standard trainer.

The loader crop covers the maximum source footprint over the permitted rotations
and scaling, with two extra pixels per crop dimension. It includes the intermediate
angles missed by the legacy crop helper. For a 512 x 512 network patch, this gives
a 1451 x 1451 loader crop. This uses more host memory; the network input size is
unchanged. Padding beyond the acquired image boundary is still necessary.

The trainer accepts only 2D configurations. It does not change spacing metadata or
replace the normal nnU-Net preprocessing, so stage the corrected image/label pairs
before preprocessing. Augmentation settings and crop sizes are logged at startup.

Results use `nnUNetTrainerGuideCatheterV3__nnUNetPlans__2d/fold_all`. Use the same
`-tr nnUNetTrainerGuideCatheterV3` option with `nnUNetv2_predict`, and keep this
trainer installed for inference. For fold `all`, evaluate `checkpoint_final.pth`
on the separately held-out cases.

CPU-only checks (no dataset, model initialization, or training):

```bash
python -m unittest nnunetv2.tests.test_guide_catheter_v3 -v
```
