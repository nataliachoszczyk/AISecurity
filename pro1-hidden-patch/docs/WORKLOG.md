# Worklog - Project 1

## 2026-10-31

- Made the project folder and the shared files of the repository (`.gitignore`, `requirements.txt`, main README).
- Added the task text in Polish (`task.md`) and in English (`task_eng.md`).
- Added the EDA: `src/eda.py` (logic) and `notebooks/01_eda.ipynb` (results). Added `scipy` to `requirements.txt`.
- Train set: all patches and all decoys are in the bottom half of the image (y_min >= 32).
- Test set: a 3x3 median on one half of the image changes the prediction. In the test set, the top half causes the change in approximately the same number of images as the bottom half. In the train set, the top half almost never causes it. Thus approximately half of the test patches are probably in the top half.
- The total strength of the attacks in the test set is approximately the same as in the train set.
- Added the consequences for training to the EDA conclusions: flip augmentation during training (vertical and horizontal, p = 0.5, fixed seed), box change with the flip, no color or noise augmentation.

## 2026-10-31 (evaluation and first method)

- Added `src/config.py` (seed, val fraction, experiment configs in `configs/*.yaml`) and `pyyaml` to `requirements.txt`.
- Added `src/evaluate.py`: the score function of the server, a report for each image kind and attack, the train/val split and the evaluation sets.
- Made the split once: `splits/split.csv`, 4500 train and 500 val, stratified by image kind and attack, seed 0.
- Checked val_flip: after a vertical flip, smoothing of the patch gives the true class for 7% of the patch images (93% without the flip). ResNet-18 misclassifies upside-down images, so val_flip does not show the work of a method on patches in the top half.
- Added `src/saliency.py`: integrated gradients toward a smoothed image give candidate boxes, and a causal check with ResNet-18 selects the patch. The method needs no training.
- Added `kaggle-jobs/` with `JOBS.md` (job list and status) and the job scripts. The smoke test found that Kaggle runs Python 3.13.
- Ran the job `saliency` on Kaggle (2x T4, 22 min). Val: saliency-v1 (median 3x3) 0.530, saliency-v2 (Gaussian blur) 0.525. Clean 0.88, decoy-only 0.94, patch images 0.30-0.51 (MI-FGSM is the worst).
- The weak part is the box: the best of the 32 candidates has a mean IoU of only 0.58 on patch images. Maps with only the positive part of the gradients were worse (0.35).
- Made `submissions/submission_saliency-v1.csv` for the leaderboard: 20.9% of the answers are 0,0,0,0, and 51% of the boxes are in the top half.
- Added `src/unet.py`: a UNet on the image and its noise residual predicts the masks of the patch and the decoys. The causal check with ResNet-18 selects the patch. Flips and 90 degree rotations are the augmentation, and the prediction is the mean over all 8.
- unet-v1 (image and residual only): val 0.350, mask IoU 0.21. The UNet does not find the weak noise well from 4000 images.
- unet-v2 (with the integrated gradients map as input): val 0.508, mask IoU 0.33. Detection is almost perfect (clean 1.00, decoy-only 0.98), but the boxes are worse than in saliency-v1 (0.530).
- Changed `kaggle-jobs/wait.sh`: it writes each status change of a job into `JOBS.md`.
- Added the small outputs of each Kaggle run to `kaggle-jobs/<job folder>/<codename>/`: `metrics.json`, `report_*.csv` and the train loss. Submission files stay out of git, because the repository is public.
- `src/unet.py` now writes the loss of each epoch to `train_log.csv`.

### Conclusions on 2026-10-31

- The best method is saliency-v1 (val 0.530). The file `submission_saliency-v1.csv` is ready, but nobody sent it to the leaderboard yet. Git does not keep it. Download it from the output of the Kaggle notebook `bumas-p1-saliency`, or run the job again.
- Detection (patch or no patch) is good in all methods. The unet-v2 detection is almost perfect (clean 1.00, decoy-only 0.98).
- The box is the weak part. MI-FGSM patches get the lowest points in all methods (0.21-0.32).
- The UNet loss still decreased at epoch 60 (`kaggle-jobs/2026-10-31_0[34]_unet-v[12]/unet-v*/train_log.txt`). Thus the training was too short.
- val_flip is not valid (see above). At this time we have no valid check for patches in the top half. The UNet is equal for top and bottom by design (8 flips and rotations in training and prediction). The saliency method has no training.

### Next steps

1. unet-v3: the same as unet-v2, but with more epochs (for example 200) and a check of the loss on the tune part.
2. Combine the methods: detection from unet-v2, box from the best of the saliency and UNet candidates.
3. Generate attacks with ResNet-18 (PGD, MI-FGSM, C&W, with a mask and JPEG) on clean and decoy-only images. Use bases from the train part for training data. Use bases from the val part for a valid val set with patches in the top half and in the middle.
4. Decide if the submission file must also go out of the git history (commit 9776e14). This needs a force push.

## 2026-10-31 (unet-v3)

- Renamed the configs to `configs/YYYY-MM-DD_NN_<codename>.yaml`. The date and NN are the same as in the name of the job folder, so the files show the order of the runs. `load_config` finds the file from the codename.
- Split the job folder `2026-10-31_03_unet` into `2026-10-31_03_unet-v1` and `2026-10-31_04_unet-v2`. The two runs were different versions of one Kaggle notebook, but they are different experiments. Each experiment now has its own number.
- `src/unet.py` now calculates the loss on the tune part after each epoch. `train_log.csv` has the new column `tune_loss`. If the tune loss increases while the train loss decreases, the UNet overfits.
- Added unet-v3: the same as unet-v2, but with 200 epochs (unet-v2 had 60).
- Added the job `2026-10-31_05_unet-v3`. From now on, the jobs run on the Kaggle account `jankombajn`. The script stops at once if the kernel has no GPU.
