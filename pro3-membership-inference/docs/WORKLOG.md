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
- Job `eda-notebook` is complete. `notebooks/01_eda.ipynb` has all results and the conclusions.

## 2026-11-28

- Made the branch `pro3-experiments-final` from `pro3`. It has only the code and the jobs of the submission on the public leaderboard: attacks-v8a (public score 0.774).
- The branch `pro3-experiments` has the full history of the experiments. The jobs below ran on that branch.
- Added the jobs of the submission, in the order of the run:
  1. `2026-11-26_05_qwen-weights` (CPU): the reference model Qwen3-0.6B-Base.
  2. `2026-11-26_08_logprobs` (GPU T4 x2): the per-token values of BUMAS-mini and Qwen3-0.6B-Base.
  3. `2026-11-26_15_refmodel` (GPU T4 x2): the two domain reference models.
  4. `2026-11-26_16_shadow` (GPU T4 x2): the shadow pair `shadow`.
  5. `2026-11-27_22_attacks-v8a` (CPU): the features, the validation and `submission_v8.csv`.
- Jobs 2 to 4 can run at the same time. Job 5 reads the outputs of jobs 2 to 4 as kernel sources.
- Changed `BRANCH` in the `run.py` of jobs 2 to 5 to `pro3-experiments-final`.
- Added `src/logprobs.py` without change.
- Added `src/refmodel.py` in the version of the commit that ran the job `shadow` (without the photo shadow models).
- Added `src/attacks.py` with only the functions of `explore_v8`. The code of the functions has no change.
- Added `docs/RESULTS.md` with the row of attacks-v8a.
