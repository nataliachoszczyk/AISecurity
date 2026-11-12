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
| 2026-11-12 | scan-smoke | `2026-11-12_02_scan-smoke` | [bumas-p2-scan-smoke](https://www.kaggle.com/code/jeronimoo/bumas-p2-scan-smoke) | T4 x2. Smoke test of the probe-bank scan: the 6 train models, 4 test models, 4 models of set A and 4 of set B. Check the code and the time for each model. Version 1: the code works, 26 to 27 s for each model. The 4 validation models of each set were clean, so no AUC there. The 6 train models (population of only 10 models): `hit_mean_top3` and `both` 1.00, `dlogp_mean` 0.98. | complete |
| 2026-11-12 | scan | `2026-11-12_03_scan` | [bumas-p2-scan](https://www.kaggle.com/code/jeronimoo/bumas-p2-scan) | T4 x2. Probe-bank scan of all 206 task models and 40 models of each validation set (A, B), then the scores and the checks. Version 1 (72 minutes): the train models give 1.00 for most scores. Set A: `hit_max_top3` 0.994, `hit_mean_top3` 0.989. Set B: `dlogp_max_rel` 0.999, `hit_max_top3` 0.992. Clean-label: A 0.995, B 0.97 to 0.99. The old detector got A 0.966 and B 0.948. | complete |
| 2026-11-12 | score-scan | `2026-11-12_04_score-scan` | [bumas-p2-score-scan](https://www.kaggle.com/code/jeronimoo/bumas-p2-score-scan) | CPU. The scores of the job `scan` with the combined scores `combo4` and `combo2` (rank mean and z mean), and the submission files. Version 1: `combo4_z` A 0.991, B 0.9998, train 1.00. `combo2_rank` A 0.995, B 0.9997. `combo4_z` gives 117 pairs above 10 in 86 test models. | complete |
| 2026-11-12 | invert-smoke | `2026-11-12_05_invert-smoke` | [bumas-p2-invert-smoke](https://www.kaggle.com/code/jeronimoo/bumas-p2-invert-smoke) | T4 x2. Smoke test of the optimized triggers (etap 3): the 6 train models, 4 test models, 4 models of set A and 4 of set B. Check the code and the time for each model. Version 1: the code works, 55 to 69 s for each model (10 minutes). | complete |
| 2026-11-12 | invert | `2026-11-12_06_invert` | [bumas-p2-invert](https://www.kaggle.com/code/jeronimoo/bumas-p2-invert) | T4 x2. Optimized triggers for the same 286 models as the job `scan`. | running |
| 2026-11-12 | score-all | `2026-11-12_07_score-all` | [bumas-p2-score-all](https://www.kaggle.com/code/jeronimoo/bumas-p2-score-all) | CPU. The scores of the probe bank together with the optimized triggers (jobs `scan` and `invert`), and the submission files. | waiting for another job |
