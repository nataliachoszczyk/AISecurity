# Worklog - Project 2

## 2026-11-09

- Made the project folder with `docs/`, `jobs/` and `src/`.
- Added the task text in Polish (`docs/task/task.md`) and in English (`docs/task/task_eng.md`).
- Changed `jobs/kaggle.sh`: it uses `.kaggle/access_token` and `.venv` from the root folder of the repository.
- The Kaggle account of the project is `jeronimoo`.

## 2026-11-10

- Decision: the task data does not go to the local machine. A Kaggle CPU job gets it from Google Drive.
- Added the job `2026-11-10_01_data`. Its output becomes the private dataset of the project.
- Checked the file list of the Drive folder: 610 files (200 test models, 6 train models, 400 clean images, 4 other files).
- The job `data` failed at first, because the account `jeronimoo` had no phone verification and the kernel had no internet.
- Version 2 of the job `data` is complete: all 610 files (4.62 GB) are in the kernel output.
- The private dataset `jeronimoo/bumas-p2-backdoors` has the output of the job `data`: 610 files, 4.62 GB. The account `jankombajn` is a reader of it.
- Decision: `model.py` of the task stays in the dataset. The repository has no copy of it.
- Added `src/paths.py`. It finds the task data and the output folder on Kaggle and on the local machine. It also loads a model with `model.py` from the dataset.
- The folder `example/` is also in the local `data/` folder (not in git). It shows the train labels: three models with one backdoor each (ASR 0.964, 0.999 and 0.581) and three clean models. The clean accuracy is 0.84 to 0.85 for all six models.

## 2026-11-12

- Started the branch `pro2-experiments-v2` from `pro2`. The work of the branch `pro2-experiments` stops. This branch takes from it only `.gitignore` and the facts below.
- Facts from the first approach:
  - A detector with trigger inversion for each pair of classes got a public AUC of 0.906. The baseline (Neural Cleanse) gets 0.901.
  - Our own validation models (finetuned from the 3 clean train models) gave 0.95 to 0.97 on the same detector. Thus they do not predict the leaderboard.
  - The Kaggle datasets `jeronimoo/bumas-p2-val-models-dataset` and `jeronimoo/bumas-p2-val-models-b-dataset` have these validation models with labels. They can only reject a method.
  - One model takes approximately 150 s for the old detector on a T4.
- Plan: for each class, the maximum over trigger candidates of the mean effect over the images. Each score is compared with the same class in all task models.
