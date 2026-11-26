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

## 2026-11-26 (branch `pro3-experiments`)

- Made the branch `pro3-experiments` from `pro3`. The branch `pro3` gets no new work until the end of the project.
- Job `model-code`: copied `openbumas.py` and the model card from the dataset, to read the API of the model. The files are not in git.
- Job `qwen-weights`: downloaded the reference model `Qwen/Qwen3-0.6B-Base` (revision `da87bfb`) into a kernel output. The GPU jobs read it from there.
- Added `src/logprobs.py`. For each token, it keeps log p of the true token and the mean and the standard deviation of log p over the vocabulary. The passes:
  - texts with BUMAS-mini and with Qwen, in windows of 2048 tokens with a step of 1024, and with EOT before the first token (as in the packed S1 pieces),
  - captions with BUMAS-mini and the own photo (the S3 format), with the photo of a different record, with two levels of photo noise, and without a photo,
  - captions with Qwen, without a photo.
- Job `logprobs-smoke` (GPU, 10 texts and 20 photos of each source type): complete, no error.
- Added `src/attacks.py`: the MIA features of each record (LOSS, zlib, Min-K%, Min-K%++, the reference attack and its variants, the gain from the own photo, the effect of the noise), owner means, and one logistic regression for each source type.
- Job `logprobs` (all 10 000 records): complete in 91 minutes.
- Jobs `attacks-v1` and `attacks-v2`: the reference attack (BUMAS-mini minus Qwen) separates the members of the text sets well (0.83 to 0.96). Forum posts give 0.48, photographers 0.34, web photos 0.11. More features of the token gains and a model of boosted trees give no gain.
- Owner check: some owners with 8 known records (all non-members) have test records with high scores. Thus some owners are victims only in test, as in dataset inference (Maini et al.).
- Jobs `attacks-v3` to `attacks-v5`: an owner term (the mean score of the other records of the owner) and a two-stage logistic regression. They help the photographers a little. The validation on known cannot show the full effect, because known has few non-members of owners that are not victims.
- Added the reference model for the domain (`src/refmodel.py`, job `refmodel`): Qwen3-0.6B-Base trained on the known non-members of one fold, two folds as a cross-fit.
- Added two shadow models (job `shadow`): Qwen3-0.6B-Base trained for one epoch on one random half of all records. Each record is IN for one model and OUT for the other.
- Job `attacks-v6`: TPR@5%FPR of the record score on known: 0.63 (v1: 0.47). Forum posts 0.84, writers 0.92, research groups 1.00, podcasts 1.00, photographers 0.50, web photos 0.11. The best feature is the gain of BUMAS-mini over the OUT shadow model.
- Added the photo shadow models (job `photo-shadow`, 4 half splits, 8 models in the S3 format) and a second pair of text shadow models (job `shadow-b`).
