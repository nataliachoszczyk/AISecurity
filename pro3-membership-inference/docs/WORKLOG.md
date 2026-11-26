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

## 2026-11-27 (branch `pro3-webphoto`, module for the web photos)

Research summary for the web photos. A web photo is one image with a short caption (approximately 24 tokens). S3 used each member one time.

- Li et al. 2024 (VL-MIA, arXiv:2411.02902): MaxRényi-K%, a target-free entropy of the logits on the image positions and on the text positions. AUC 0.70 to 0.80 on VL-MIA, but these sets have a shift of the data.
- Revisiting Data Auditing in LVLMs (arXiv:2504.18349): a blind image classifier gets AUC 0.99 on VL-MIA/Flickr. On sets without a shift, all logit attacks give AUC 0.45 to 0.57. Probes on the hidden states give up to 0.65.
- OpenLVLM-MIA (arXiv:2510.16295): a controlled benchmark of the stages of LLaVA. All attacks give AUC 0.41 to 0.53 and TPR@5%FPR below 0.08.
- Newer black-box attacks (CSA-MIA arXiv:2605.17341, DistractMIA arXiv:2605.12574) use the generated text and an image-text encoder. Our access to the logits is stronger, so these attacks add little.
- Conclusion: a single caption exposure gives a weak signal. Our AUC 0.61 is already above most results on sets without a shift.

Ideas, from the high expected value and low cost to the low value and high cost:

1. Blind check (CPU): the caption words, the account name and the image file. A shift of the data gives a strong signal, as in the papers above.
2. Calibration (CPU): fewer features and a stronger regularization. If the signal stays weak, the correct output is a low log-odds near the prior, so that the web photos do not hurt the shared threshold.
3. z-scores of the features against the known non-members inside bins of the caption length (the idea of the best team for the photos).
4. Duplicate captions (CPU): the member of a group with the same caption has a higher log p than the other photos of the group.
5. Caption ranking (GPU): the log p of the caption with its own photo against the log p with 31 other photos. This gives a z-score of the own photo with a small noise, and a rank of the own caption for the photo.
6. Target-free entropy (MaxRényi) on the image positions and on the caption positions (GPU, in the same pass).
7. A linear probe on the hidden states of the image tokens and of the caption tokens (GPU pass, CPU fit). The papers show a small gain.

New file: `src/webphoto.py`. It has all code of the web photo module, so that the shared files do not change.

- Job `webphoto-v1` (CPU): ideas 1 to 4 on the passes of the earlier jobs.
- Job `webphoto-v1` is complete. Results inside the web photos (TPR@5%FPR, mean and std over 2 folds x 5 seeds):
  - The v8a features give 0.127 (0.028), AUC 0.621. Their log loss is 0.509, and the prior alone gives 0.514. Thus the log-odds are almost not better than the prior.
  - A stronger regularization (C = 0.01) gives 0.147 (0.028), AUC 0.628, log loss 0.497. The gain is smaller than the std, but the calibration is better.
  - Small feature sets, z-scores in bins of the caption length, the duplicate captions and the length give no gain.
  - A fit together with the photographers gives 0.150 (0.041), but a worse log loss (0.529), because the photographers have fewer members.
  - The blind scores (image file, caption words, account name) give AUC 0.48 to 0.52. Thus the data has no shift, as in a controlled benchmark.
  - `img_gain` (own photo against a swapped photo) has AUC 0.48: the members depend less on their photo than the non-members.
- Added `run_matrix` to `src/webphoto.py` (ideas 5 and 6). Each photo goes into a random group of 32 photos of its type. The pass scores each caption of the group with each photo of the group. It also scores the own caption with the flipped photo. For the own pair, it keeps the Renyi entropies and the mean hidden states (layers 14 and 28, image tokens and caption tokens).
- Job `webphoto-matrix-smoke` (40 photos of each type): complete, no error.
- Added `run_grad`: the gradient norms of the caption loss of the own pair for the connector, the embeddings, each LLM layer and the image tokens. Job `webphoto-grad-smoke` tests it.
- Merged `pro3-experiments` into `pro3-webphoto` for the simulation of the global metric (`explore_v9`). Added `global_sim`: it puts the web photo log-odds of a variant in place of the v8a log-odds, and keeps the rest of v8a.
