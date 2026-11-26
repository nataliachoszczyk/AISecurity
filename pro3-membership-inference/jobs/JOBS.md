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
| 2026-11-26 | eda | `2026-11-26_02_eda` | [bumas-p3-eda](https://www.kaggle.com/code/jeronimoo/bumas-p3-eda) | CPU. Data analysis of known and test: source types, members, owners, lengths, baselines without the model, example texts and photos. The output is `eda/report.html` (not in git, because it shows task data), `summary.json`, `owners.csv` and `baselines.csv`. | to run |
