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
| 2026-11-10 | data | `2026-11-10_01_data` | [bumas-p2-data](https://www.kaggle.com/code/jeronimoo/bumas-p2-data) | CPU job. Download the task data (610 files, approximately 4.3 GB) from Google Drive into the output folder. Check the internet and the clone of the repository on the account `jeronimoo`. Version 1 failed after 34 seconds: the kernel had no internet ("Could not resolve host: github.com"). The account had no phone verification. Version 2: 610 files, 4.62 GB, in 4 minutes. | complete || 2026-11-12 | scan-smoke | `2026-11-12_02_scan-smoke` | [bumas-p2-scan-smoke](https://www.kaggle.com/code/jeronimoo/bumas-p2-scan-smoke) | T4 x2. Smoke test of the probe-bank scan: the 6 train models, 4 test models, 4 models of set A and 4 of set B. Check the code and the time for each model. | to run |
| 2026-11-12 | scan | `2026-11-12_03_scan` | [bumas-p2-scan](https://www.kaggle.com/code/jeronimoo/bumas-p2-scan) | T4 x2. Probe-bank scan of all 206 task models and 40 models of each validation set (A, B), then the scores and the checks. | waiting for another job |
