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

## 2026-11-11

- The work continues on the branch `pro2-experiments`. The Kaggle jobs run on the account `jeronimoo`.
- Decision: no Neural Cleanse baseline. The models have the same base, the same data and the same finetune. Thus a comparison of a class with the same class in the other models can show a backdoor.
- Added `src/model_stats.py`. It collects statistics of the weights and of the answers of each model. It compares each class with the same class in all 206 models (z-score). The seed is 0.
- Added the job `2026-11-11_02_model-eda`. It is the EDA of the models.
- Version 1 of the job `model-eda` failed: the image id in `clean/labels.csv` has no file extension. The code now searches for `images/<id>.*`.
- Version 2 of the job `model-eda` failed at the AUC step: a z-score of a statistic with std 0 was NaN. Now such a z-score is 0. The job saves the raw tables first and catches errors of the analysis, because Kaggle deletes all output of a failed job.
