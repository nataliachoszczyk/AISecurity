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
| 2026-11-26 | data | `2026-11-26_01_data` | [bumas-p3-data](https://www.kaggle.com/code/jeronimoo/bumas-p3-data) | CPU job. Download `p3_student.zip` (approximately 1.25 GB) from Google Drive into the output folder. Check the records in the zip and the clone of the repository. Then make the dataset from the output on the web page. Version 1: complete in 1 minute. The zip has 1.25 GB and 3565 entries: 3545 images, 15 model files, `known.jsonl` and `test.jsonl` with 5000 records each, `sources.csv` and `sample_submission.csv`. The output is the private dataset `jeronimoo/bumas-p3-data-set` (folder `p3_student/`). | complete |
| 2026-11-26 | eda | `2026-11-26_02_eda` | [bumas-p3-eda](https://www.kaggle.com/code/jeronimoo/bumas-p3-eda) | CPU. Data analysis of known and test: source types, members, owners, lengths, baselines without the model, example texts and photos. The output is `eda/report.html` (not in git, because it shows task data), `summary.json`, `owners.csv` and `baselines.csv`. Version 1 was cancelled after a few minutes: it used the old metric code. Version 2 uses the copy of the server function: complete in 14 minutes. Known has 16.7% members. 12 of the 60 owners of each set type (podcast: 8 of 40) have 6 members in 14 known records (podcast: 4 in 9). All other owners have 0 known members. Length gives no signal (TPR@5%FPR 0.03 to 0.05). The owner prior in the report is not correct: the leave-one-out rate is lower for a member than for a non-member of the same owner. | complete |
| 2026-11-26 | eda-notebook | `2026-11-26_03_eda-notebook` | [bumas-p3-eda-notebook](https://www.kaggle.com/code/jeronimoo/bumas-p3-eda-notebook) | CPU. Run `notebooks/01_eda.ipynb` on the task data. The output is the notebook with all results. It replaces the HTML report of the job `eda`, and it adds the validation method (`src/validation.py`) and a correct owner prior. Version 1: complete in 5 minutes. Owner prior: AUC 0.74, TPR@5%FPR 0.019 (ties), 0.18 with a random order inside a tie. Version 2 has better figures and the conclusions: complete in 7 minutes. The run notebook is `notebooks/01_eda.ipynb`. | complete |
| 2026-11-26 | qwen-weights | `2026-11-26_05_qwen-weights` | [bumas-p3-qwen-weights](https://www.kaggle.com/code/jeronimoo/bumas-p3-qwen-weights) | CPU. Download the reference model `Qwen/Qwen3-0.6B-Base` from Hugging Face into the output (folder `qwen3-0.6b-base/`, the file `REVISION` has the commit). The GPU jobs read it as a kernel source. Version 1: complete in 2 minutes, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. | complete |
| 2026-11-26 | logprobs | `2026-11-26_08_logprobs` | [bumas-p3-logprobs](https://www.kaggle.com/code/jeronimoo/bumas-p3-logprobs) | GPU T4 x2. `src/logprobs.py` on all 10 000 records. BUMAS-mini (GPU 0): texts, captions with the own photo, with a swapped photo, with two levels of photo noise, and without a photo. Qwen3-0.6B-Base (GPU 1): texts and captions. Version 1: complete in 91 minutes, no error. 6456 texts (4.79 million tokens) and 3544 captions (154 329 tokens). The BUMAS-mini texts take 52 minutes. | complete |
| 2026-11-26 | refmodel | `2026-11-26_15_refmodel` | [bumas-p3-refmodel](https://www.kaggle.com/code/jeronimoo/bumas-p3-refmodel) | GPU T4 x2. `src/refmodel.py`: train Qwen3-0.6B-Base on the known non-members of fold 0 (GPU 0) and fold 1 (GPU 1) of `make_folds(seed=0)`, 2 epochs, LR 3e-5. Then score all texts and all captions (as text). Version 1: complete in 2 hours 22 minutes, no error. Each fold trains on approximately 2080 known non-members. | complete |
| 2026-11-26 | shadow | `2026-11-26_16_shadow` | [bumas-p3-shadow](https://www.kaggle.com/code/jeronimoo/bumas-p3-shadow) | GPU T4 x2. `run_shadow` of `src/refmodel.py`: two shadow models (LiRA). Each trains Qwen3-0.6B-Base on one random half of all records (known and test, texts and captions, no labels), one epoch, LR 3e-5. Then each scores all texts and captions. Each record is IN for one model and OUT for the other. Version 1: complete in 2 hours 45 minutes, no error. One training step takes approximately 38 seconds. | complete |
| 2026-11-27 | attacks-v8a | `2026-11-27_22_attacks-v8a` | [bumas-p3-attacks-v8a](https://www.kaggle.com/code/jeronimoo/bumas-p3-attacks-v8a) | CPU. `explore_v8` with the shadow pair of `shadow`: the record score with the v6 features (no owner term), its validation, the expected members in test for each owner group, the submission file. Version 1: complete in 1 minute. TPR@5%FPR (mean, std over 2 folds x 5 seeds): all 0.631 (0.021), forum posts 0.843, photographers 0.503 (0.066), podcasts 1.000, research groups 0.997, web photos 0.111 (0.023, AUC 0.61), writers 0.919. Expected members in test: research groups 266, writers 236, web photos 189, forum posts 178, photographers 129, podcasts 105. | complete |
