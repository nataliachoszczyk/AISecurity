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

## 2026-11-27 (coordinator, branch `pro3-experiments`)

- Started two agents on their own branches and Kaggle accounts: web photo (`pro3-webphoto`, `jankombajn`, jobs 30 to 49) and photographer (`pro3-photographer`, `nataliachoszczyk`, jobs 50 to 69).
- Job `attacks-v8a`: CV TPR@5%FPR of all known records 0.631 (std 0.021). The first submission of its file gave 0.774 on the public part.
- Added a simulation of the test metric to `src/attacks.py` (`explore_v9`). Known records get weights, so that each source type has its expected test mix of members and non-members.
- Added the owner prior: in the set types, the prior of the owner (from its other records) replaces the prior of the source type in the record log-odds.
- Job `attacks-v9`: the simulation gives 0.741 at w = 0 (public: 0.774). Thus the simulation is a usable check of the global metric.
- Web photos, forum posts and photographers give most false positives at the global threshold. The owner prior gives 0.756 (w = 1) and helps photographers and writers.

## 2026-11-27 (branch `pro3-photographer`)

The branch `pro3-photographer` comes from `pro3-experiments`. It holds the work on the photographer records. The Kaggle account is `nataliachoszczyk`, the job numbers are 50 to 69.

### Research summary

Sources:

- Maini et al. 2024 (LLM Dataset Inference): one linear map combines many MIA features, then a t-test compares the suspect set with a set of non-members.
- Puerto et al. 2024 (Scaling Up Membership Inference): the scores of many records add up. A small gain of the record AUC gives a large gain at the level of a collection.
- Li et al. 2024 (VL-MIA): MaxRényi-K%, a target-free score from the entropy of the output distribution. Best AUC on descriptions is approximately 0.71.
- Lo et al. 2026 (COMPASS): image noise at many levels, with the noise level and the direction calibrated on known non-members. AUC up to 0.88.
- Hu et al. 2025 (USENIX Security, MIA against VLMs): set-level inference on 5 samples gives AUC above 0.8.
- Miyamoto et al. 2025 (OpenLVLM-MIA): many VLM results come from a distribution shift, not from membership.

Facts from the existing results:

- In known, a victim owner has 6 members in 14 records. In test, the strong record score of research groups gives 159 expected members in the 168 test records of victim owners. Thus the test part of a victim owner is almost all members.
- The owner prior alone gives approximately 0.81 on the leaderboard (information from the coordinator). Thus most test members are in the sets of victim owners.
- The record score of attacks-v8a is calibrated to the member rate of known (12%). Thus its estimate of 129 test members for photographers is too low.

Ideas, ranked by the expected value and the cost:

1. An owner mixture model (dataset inference): each owner is a victim or clean, with a prior for each owner group (victim, full, other). EM on the unlabeled test records fits the parameters. The output is the posterior log-odds of each record. Cost: CPU only.
2. A simpler owner term: the record score plus the mean score of the other records of the owner, or a t statistic of the owner mean (as the "photo_z" file name suggests). Cost: CPU only.
3. Entropy features (MaxRényi-K% with alpha 1): the passes already keep the mean log p over the vocabulary, which is the negative entropy. Cost: CPU only.
4. Per-owner calibration of the record features (a z-score inside the owner). Cost: CPU only.
5. New GPU passes (COMPASS noise levels, generated captions). Cost: GPU hours, small expected gain, because the noise and shadow features already gave no gain.

### Design of the honest owner check

The validation on known cannot show the owner term, because in known only 6 of 14 records of a victim are members and no owner is a hidden victim. Thus the check makes a semi-synthetic test from known photographer records with the structure of the real test:

1. Compute the out-of-fold record llr of all known photographer records (2 folds, 5 seeds).
2. Put the 12 victim owners into two random halves.
3. Keep the members of the first half in the group "victim". These owners copy the test part of a victim owner (almost all members).
4. Keep the members of the second half, but put them into the group "other". These owners copy hidden victims.
5. Keep all records of the full and the other owners as clean owners (all non-members).
6. Fit the mixture with EM without labels, then measure TPR@5%FPR and AUC on all records of the semi-synthetic test.
7. Repeat for 10 random halves of each seed (50 runs).

Limits of the check:

- A semi-synthetic owner has 6 records, a real test owner has 14. Thus the check underestimates the owner evidence.
- The check supposes that the test part of a victim is all members and that a clean owner has no members. The job also estimates this structure on the test records of the text types, where the record score is almost exact.
- The number of hidden victims in the check (6 of 46 other owners) is an estimate. The real number comes from the EM on test.

### Job `photographer-v1`

- Added `src/photographer.py`, the one code file of the photographer module. It imports the features of `src/attacks.py` and changes no shared file.
- Added the job `2026-11-27_50_photographer-v1` (CPU). It does the structure check on the test of all set types, the known CV, the honest owner check and the file `photographer_logodds.csv`.
