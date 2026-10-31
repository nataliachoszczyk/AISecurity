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
