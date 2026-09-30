utils/dataloader_retina.pyutils/data.py# BEAM-Net# BEAM-Netutils/dataloader_retina.pyutils/data.py# BEAM-Net

BEAM-Net (Boundary-Enhanced Asymmetric Multi-scale Network) is the retinal vessel segmentation model described in the accompanying paper. It targets DRIVE and CHASE_DB1 and combines a ResNet-34 encoder with RMSAB, AS-MSCB, AAF, dynamic boundary enhancement, deep supervision, and Focal + Dice + clDice loss.

## Repository scope

This review build contains source code, utilities, experiment entry points, and dependency files. Dataset images, checkpoints, logs, TensorBoard files, generated predictions, and result spreadsheets are intentionally excluded. Place datasets under `data/DRIVE` and `data/CHASE_DB1` locally, following the loader's expected `train`, `val`, and `test` layout.

## Main entry points

- `train.py`: training and validation with the paper's optimizer and loss weights.
- `test_final.py`: standalone evaluation; threshold search uses the validation split only.
- `run_ablation.py`: ablation experiment launcher.
- `lib/networks.py`: BEAM-Net model wrapper and prediction heads.
- `lib/decoders.py`: AS-MSCB, RMSAB/GRAB, AAF, EUCB, and LGAG components.
- `lib/resnet.py`: ResNet encoder with standard convolutions.
- `utils/dataloader_retina.py`: DRIVE/CHASE_DB1 loader and augmentation pipeline.

The default training configuration follows the paper: ResNet-34, 512px input, batch size 4, AdamW, learning rate 1e-3, weight decay 1e-4, 200 epochs, 10 warm-up epochs, minimum learning rate 1e-6, AMP on CUDA, warm-up plus cosine decay, and deep supervision. Run `python train.py --help` before training and provide a local dataset path when needed.

Example: `python train.py --data-root D:/datasets/DRIVE --dataset DRIVE --no-pretrain`. Evaluation defaults to the paper's validation-calibrated threshold and test-time augmentation; use `--no-tta --fixed-threshold` for the ablation protocol.

## Reproducibility notes

The paper reports DRIVE and CHASE_DB1 splits, metrics, and ablation results. The source directories contain several historical variants; the selected baseline is the most complete `xiaorongshiyan` version, with the enhanced `resnet` and FOV handling reconciled as documented in `docs/PROJECT_REVIEW.md`.

This code is released for research review. Dataset and pretrained-weight licenses remain the responsibility of the user.

