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
| 2026-11-10 | data | `2026-11-10_01_data` | [bumas-p2-data](https://www.kaggle.com/code/jeronimoo/bumas-p2-data) | CPU job. Download the task data (610 files, approximately 4.3 GB) from Google Drive into the output folder. Check the internet and the clone of the repository on the account `jeronimoo`. Version 1 failed after 34 seconds: the kernel had no internet ("Could not resolve host: github.com"). The account had no phone verification. Version 2: 610 files, 4.62 GB, in 4 minutes. | complete || 2026-11-11 | model-eda | `2026-11-11_02_model-eda` | [bumas-p2-model-eda](https://www.kaggle.com/code/jeronimoo/bumas-p2-model-eda) | GPU job. EDA of all 206 models: the distance of each layer to the mean model, a sketch of the weight changes, the output layer, and the answers to clean images, noisy images, blended images and random noise. Writes CSV and PNG files and the AUC of each statistic on the train pairs. No separate smoke test: the job is short. | to run |
