# Worklog - Project 1

## 2026-10-31

- Made the project folder and the shared files of the repository (`.gitignore`, `requirements.txt`, main README).
- Added the task text in Polish (`task.md`) and in English (`task_eng.md`).
- Added the EDA: `src/eda.py` (logic) and `notebooks/01_eda.ipynb` (results). Added `scipy` to `requirements.txt`.
- Train set: all patches and all decoys are in the bottom half of the image (y_min >= 32).
- Test set: a 3x3 median on one half of the image changes the prediction. In the test set, the top half causes the change in approximately the same number of images as the bottom half. In the train set, the top half almost never causes it. Thus approximately half of the test patches are probably in the top half.
- The total strength of the attacks in the test set is approximately the same as in the train set.
- Added the consequences for training to the EDA conclusions: flip augmentation during training (vertical and horizontal, p = 0.5, fixed seed), box change with the flip, no color or noise augmentation.
