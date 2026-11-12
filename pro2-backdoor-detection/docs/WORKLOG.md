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
- Decision (user): the full plan runs without a question for each job.
- Added `src/scan.py`. It applies a fixed bank of 196 trigger candidates (probes) to the 400 clean images: patches, blends, SIG stripes, warps, color changes, quantization, blur, noise, high and low frequencies, flips and full replacements. For each probe, source class and class, it keeps the mean change of the log probability and the fraction of images that go to the class.
- Added `src/score.py`. It compares each value with the same value in all models of the set (robust z-score). The score of a class is the maximum over the probes of the mean reaction over the source images. A second variant takes the maximum over the source classes for a targeted backdoor.
- The checks: the AUC on the 6 train models and on 40 models of each validation set (only to reject a method), and a count of the outliers in the test models (the task suggests approximately 150 positive pairs in approximately 100 models).
- Added the jobs `2026-11-12_02_scan-smoke` and `2026-11-12_03_scan`.
- The job `scan` is complete (72 minutes on T4 x2, 27 s for each model).
  - The 6 train models: AUC 1.00 for most scores (only 3 positive pairs).
  - Set A: `hit_max_top3` 0.994, `hit_mean_top3` 0.989, `dlogp_mean` 0.947. Clean-label (SIG) 0.995.
  - Set B: `dlogp_max_rel` 0.999, `both` 0.994, `hit_max_top3` 0.992. Clean-label 0.97 to 0.99.
  - The old detector got A 0.966 and B 0.948 on the same sets. The fraction of images that go to the class (`hit`) is better than the change of the log probability (`dlogp`) on set A.
  - Test models: `hit_mean_top3` gives 110 pairs above 10 in 87 models. The task suggests approximately 150 pairs in approximately 100 models.
- Added the combined scores `combo4` (`hit_max_top3`, `hit_mean_top3`, `dlogp_max_rel`, `dlogp_mean_rel`) and `combo2` (`hit_max_top3`, `dlogp_max_rel`): the mean of the ranks and the mean of the z-scores.
- Added the CPU job `2026-11-12_04_score-scan`. It uses the features of the job `scan` and makes the submission files.
- The job `score-scan` is complete. The combined scores are the best on the sets A and B:
  - `combo4_z`: A 0.991, B 0.9998, train 1.00. Test models: 117 pairs above 10 in 86 models.
  - `combo2_rank`: A 0.995, B 0.9997, train 1.00.
- Decision: the first submission of the new detector is `scan_combo4_z.csv`. It uses 4 parts, so it depends less on one type of trigger. The z mean keeps a strong signal of one part, and a rank mean does not.
- Decision: the optimized triggers (etap 3) start now, without the public score. They protect against trigger types that are not in the bank, for example in the private part of the test set. The sets A and B can only show that they do not decrease the score.
- Added `optimize_triggers()` to `src/scan.py`. For each target class, it optimizes one trigger on 3 images of each other class (25 steps) and measures it on the other 17 images of each class. There are 2 families: additive (at most 0.05) and patch (mask and pattern, with a penalty on the mask size). The results have the same form as the probes, so `src/score.py` uses them as 2 more probes.
- Added the jobs `2026-11-12_05_invert-smoke` and `2026-11-12_06_invert`.
