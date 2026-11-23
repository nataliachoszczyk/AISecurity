# Jobs

All Kaggle jobs of the project. The full rules are in `KAGGLE.md`.

## Status values

Use only these values in the column Status. `wait.sh` changes the status cell, so the cell has one value and no other text.

| Status | Meaning |
|---|---|
| `to run` | The job is ready, but it did not start. |
| `waiting for another job` | The job starts after the end of a different job. |
| `queued` | Kaggle has the job, but it did not start. |
| `running` | The job runs. |
| `complete` | The job ended without an error. |
| `failed` | The job ended with an error. |
| `cancelled` | A person stopped the job, or the team does not run it. The comment gives the reason and the partial results. |

Put the results and the notes into the column Comment.

## Table

| Date | Job | Folder | Link | Comment | Status |
|---|---|---|---|---|---|
| 2026-11-10 | data | `2026-11-10_01_data` | [bumas-p2-data](https://www.kaggle.com/code/jeronimoo/bumas-p2-data) | CPU job. Download the task data (610 files, approximately 4.3 GB) from Google Drive into the output folder. Check the internet and the clone of the repository on the account `jeronimoo`. Version 1 failed after 34 seconds: the kernel had no internet ("Could not resolve host: github.com"). The account had no phone verification. Version 2: 610 files, 4.62 GB, in 4 minutes. | complete |
| 2026-11-11 | val-models | `2026-11-11_02_val-models` | [bumas-p2-val-models](https://www.kaggle.com/code/jeronimoo/bumas-p2-val-models) | T4 x2. Make the 120 models of the validation set A with known backdoors (60 clean, 35 with 1, 17 with 2, 8 with 3 backdoors) on ImageNet 64x64 (dataset `j53t3r/imagenet64x64`). The output is the dataset `jeronimoo/bumas-p2-val-models-dataset`. Version 1 was cancelled after 55 minutes: the vote class map used wrong images for Walker_hound and ashcan. Version 2 uses the class map of the ILSVRC devkit: complete in 2.9 hours; 114 of 120 models are valid (60 clean, 34 with 1, 13 with 2, 7 with 3 backdoors). Mean ASR: 0.72 (SIG) to 1.00 (patch). Clean accuracy: 0.86 (clean models) to 0.78 (2 or 3 backdoors). The job ran from the branch `pro2-experiments`. | complete |
| 2026-11-12 | val-models-b | `2026-11-12_03_val-models-b` | [bumas-p2-val-models-b](https://www.kaggle.com/code/jeronimoo/bumas-p2-val-models-b) | T4 x2. Make the 80 models of the validation set B with other trigger types (40 clean, 24 with 1, 11 with 2, 5 with 3 backdoors). The output is the dataset `jeronimoo/bumas-p2-val-models-b-dataset`. Version 1: complete in 2.4 hours; 69 of 80 models are valid (40 clean, 21 with 1, 6 with 2, 2 with 3 backdoors). Most failures have a clean-label patch. The job ran from the branch `pro2-experiments`. | complete |
| 2026-11-12 | scan | `2026-11-12_04_scan` | [bumas-p2-scan](https://www.kaggle.com/code/jeronimoo/bumas-p2-scan) | T4 x2. Probe-bank scan of all 206 task models and 40 models of each validation set (A, B), then the scores and the checks. Version 1 (72 minutes): the train models give 1.00 for most scores. Set A: `hit_max_top3` 0.994, `hit_mean_top3` 0.989. Set B: `dlogp_max_rel` 0.999, `hit_max_top3` 0.992. Clean-label: A 0.995, B 0.97 to 0.99. The old detector got A 0.966 and B 0.948. | complete |
| 2026-11-12 | score-scan | `2026-11-12_05_score-scan` | [bumas-p2-score-scan](https://www.kaggle.com/code/jeronimoo/bumas-p2-score-scan) | CPU. The scores of the job `scan` with the combined scores `combo4` and `combo2` (rank mean and z mean), and the submission files. Version 1: `combo4_z` A 0.991, B 0.9998, train 1.00. `combo2_rank` A 0.995, B 0.9997. `combo4_z` gives 117 pairs above 10 in 86 test models. | complete |
| 2026-11-23 | readme-figures | `2026-11-23_06_readme-figures` | [bumas-p2-readme-figures](https://www.kaggle.com/code/jeronimoo/bumas-p2-readme-figures) | CPU. Draw the figures of the README: the classes, all 196 probes (one figure for each family), the triggers of the sets A and B, examples of the scan and the results. It uses the features of the job `scan` and the labels of the sets A and B. Version 1: complete in approximately 2 minutes. Version 2: the clean-label types of set B are marked in the figures. Versions 3 and 4: no rows with the change, simpler scan figures. The PNG files are in `assets/`. | complete |
