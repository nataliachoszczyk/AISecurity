# Worklog - Project 3

## 2026-11-23

- Made the project folder with `assets/`, `docs/`, `jobs/`, `src/` and `submissions/`.
- Added the task text in Polish (`docs/task/task.md`) and in English (`docs/task/task_eng.md`).
- Added the two figures of the task (`assets/pipeline.svg`, `assets/tpr_fpr.svg`).
- Copied `jobs/kaggle.sh` and `jobs/wait.sh` from project 2. Added `jobs/JOBS.md` with an empty table.

## 2026-11-26

- Added the job `2026-11-26_01_data` (CPU, account `jeronimoo`). It downloads `p3_student.zip` from Google Drive on Kaggle and checks the records. The local machine does not download the data.
- Made the private dataset `jeronimoo/bumas-p3-data-set` from the output of the job `data`.
- Added `src/data.py` (load the records), `src/metrics.py` (TPR@5%FPR as on the server, checked against the server function) and `src/eda.py` (HTML report).
- Added the job `2026-11-26_02_eda` (CPU). It makes the data analysis report on Kaggle.
- Added a `.gitignore` line for HTML outputs, because the report shows task data.
- Job `eda` is complete. The report is in `jobs/runs/2026-11-26_02_eda/output/eda/report.html` (local only). `summary.json` and `baselines.csv` are in git.
- Main result: only 12 of 60 owners of each set type (podcast: 8 of 40) have known members, 6 of 14 records each (podcast: 4 of 9). Length gives no signal.
- Changed `src/eda.py`: it gives figures and tables for the notebook `notebooks/01_eda.ipynb`, not an HTML report.
- Added `src/validation.py`: 2 folds that copy the relation between known and test (split inside each set owner), 5 seeds, the server metric for each fold.
- Added the job `2026-11-26_03_eda-notebook` (CPU). It runs the notebook on Kaggle.
