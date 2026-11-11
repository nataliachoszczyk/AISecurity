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
| 2026-11-10 | data | `2026-11-10_01_data` | [bumas-p2-data](https://www.kaggle.com/code/jeronimoo/bumas-p2-data) | CPU job. Download the task data (610 files, approximately 4.3 GB) from Google Drive into the output folder. Check the internet and the clone of the repository on the account `jeronimoo`. Version 1 failed after 34 seconds: the kernel had no internet ("Could not resolve host: github.com"). The account had no phone verification. Version 2: 610 files, 4.62 GB, in 4 minutes. | complete || 2026-11-11 | model-eda | `2026-11-11_02_model-eda` | [bumas-p2-model-eda](https://www.kaggle.com/code/jeronimoo/bumas-p2-model-eda) | GPU job. EDA of all 206 models: the distance of each layer to the mean model, a sketch of the weight changes, the output layer, and the answers to clean images, noisy images, blended images and random noise. Writes CSV and PNG files and the AUC of each statistic on the train pairs. No separate smoke test: the job is short. Version 1 failed after 15 seconds: the image id in `clean/labels.csv` has no file extension. Version 2 failed after 10 minutes at the AUC step: NaN in a z-score (std 0). Version 3: complete in 9 minutes. Best statistic on the train pairs: z_clean_logit (AUC 0.96, target ranks 1, 1, 7). The weights show no difference between clean and poisoned models. | to run |
| 2026-11-11 | val-models-smoke | `2026-11-11_03_val-models-smoke` | [bumas-p2-val-models-smoke](https://www.kaggle.com/code/jeronimoo/bumas-p2-val-models-smoke) | GPU job. Smoke test of `val-models`: the first 12 validation models. Version 1: complete, but the clean accuracy fell to 0.60 (also for clean models), and the warp trigger did not work (ASR 0.10 to 0.36 after 4 attempts). Version 2 (BN eval, lr 0.002, distillation): all models give one answer (accuracy 0.05). Version 3: diagnosis of 3 finetune settings on 3 models. | complete |
| 2026-11-11 | val-models | `2026-11-11_04_val-models` | [bumas-p2-val-models](https://www.kaggle.com/code/jeronimoo/bumas-p2-val-models) | GPU job. Make 120 validation models with known backdoors (60 clean, 35 with 1, 17 with 2, 8 with 3 backdoors). The output becomes the dataset `jeronimoo/bumas-p2-val-models`. | waiting for another job |
