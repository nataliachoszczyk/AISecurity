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
| 2026-11-26 | data | `2026-11-26_01_data` | [bumas-p3-data](https://www.kaggle.com/code/jeronimoo/bumas-p3-data) | CPU job. Download `p3_student.zip` (approximately 1.25 GB) from Google Drive into the output folder. Check the records in the zip and the clone of the repository. Then make the dataset from the output on the web page. | to run |
