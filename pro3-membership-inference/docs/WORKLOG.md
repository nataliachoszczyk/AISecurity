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
- Job `photographer-v1` is complete in 4 minutes. Results:
  - The test structure of the text types is clear. Each type has 12 victims with all test records as members, and 8 hidden victims among the 40 other owners, also all members. The 8 full owners have no members. Thus each type has 280 test members.
  - On the photographer test, the EM finds q_other 0.21 (8.5 hidden victims), r 0.88 and 251 expected members. The record score of attacks-v8a gives only 129.
  - Known CV (other folds than the baseline): record 0.458 (std 0.105), record with the prior of the owner group 0.594 (std 0.053).
  - Honest check (TPR@5%FPR, mean and std of 50 runs): record 0.631 (0.065), group prior alone 0.831 (0.045), mixture tau 1 0.972 (0.040), mixture tau 0.5 0.978 (0.037), owner mean w 3 0.956 (0.032).
- The file `photographer_logodds.csv` has the ids and scores of test records. The repository is public, thus the file name now starts with `submission_` and git ignores it.
- An incident: all worktrees share one git stash list. A `git stash pop` took the stash of the web photo agent. It was put back, and nothing was lost. Do not use `git stash` in a worktree.
- Merged `origin/pro3-experiments` (the global simulation `explore_v9` and `owner_prior_shift` of the coordinator).
- Changed `src/photographer.py` after the comments of the coordinator:
  - New scenarios of the honest check break the assumptions of the mixture. Each scenario changes one item:
    - victims with 75% members,
    - clean owners with approximately 4% members,
    - 3 or 12 hidden victims,
    - all three errors together.
  - The check compares the mixture with `owner_prior_shift(w=1)`, the baseline of the coordinator.
  - `global_sim` runs the global simulation of `explore_v9` with the mixture log-odds for the photographer records.
  - `set_logodds` gives the log-odds of any set type, so that the coordinator can also use it on the text types.
  - Removed `explore_v1`, `owner_z` and the owner mean with w 1. The job `photographer-v1` ran the commit `f3c7b19`.
- Added the job `2026-11-27_51_photographer-v2` (CPU). Record variants, one change each against v6: C 0.1, the entropy features (MaxRényi-K% with alpha 1, gains over the OUT photo shadow models), the PLIRA features.
- Job `photographer-v2` is complete in 6 minutes. TPR@5%FPR (mean, std):

  | Record variant | Known CV record | Known CV + group | Check: record | Check: mixture tau 1 | Check: prior shift w 1 |
  |---|---|---|---|---|---|
  | v6 | 0.503 (0.066) | 0.597 (0.066) | 0.650 (0.034) | 0.982 (0.034) | 0.817 (0.041) |
  | v6 C 0.1 | 0.531 (0.062) | 0.639 (0.037) | 0.722 (0.018) | 1.000 (0.000) | 0.867 (0.026) |
  | v6 + entropy | 0.469 (0.093) | 0.631 (0.054) | 0.642 (0.054) | 0.953 (0.040) | 0.797 (0.037) |
  | v6 + PLIRA | 0.478 (0.070) | 0.578 (0.057) | 0.594 (0.063) | 0.944 (0.050) | 0.753 (0.041) |

  - The entropy features are weak alone (best: Min-K%++ gain over the OUT photo shadow models, AUC 0.74). They and PLIRA make the record score worse. Thus they are removed.
  - C 0.1 is better. On test it gives r 0.95, q_other 0.22 and 277 expected members (the structure of the text types gives 280). Nine other owners have an owner posterior above 0.9, the next one has 0.03.
  - Scenarios with a wrong structure (v6, check: mixture tau 1, tau 0.5, prior shift w 1): victims with 75% members 0.835, 0.886, 0.695. Clean owners with 4% members 0.808, 0.754, 0.726. 3 hidden victims 0.992, 0.997, 0.817. 12 hidden victims 0.950, 0.967, 0.817. All three errors together 0.692, 0.700, 0.652 (group prior alone: 0.749).
  - Thus the mixture stays better than the prior shift in all scenarios. With all errors together, the difference is smaller than the std.
  - Global simulation (`explore_v9` style, v6): 0.769 (std 0.017) with the mixture for the photographer records, 0.756 (0.019) with the prior shift w 1. Photographer TPR at the global threshold: 0.872 against 0.647.
- Removed the entropy features from `src/photographer.py`.
- Added the job `2026-11-27_52_photographer-v3` (CPU): C 0.1, 0.03 and 0.01, and v6 without the domain reference features (C 0.1).
  - The cross-fit of the domain features differs between known and test. A known record gets the model of the other fold, a test record gets the mean of the two models.
  - The job also writes the llr distributions of known and test (`llr_shift`).
- Job `photographer-v3` is complete in 5 minutes. TPR@5%FPR (mean, std):

  | Record variant | Known CV record | Known CV + group | Check: record | Check: mixture | Test: expected members |
  |---|---|---|---|---|---|
  | v6 C 0.1 | 0.531 (0.062) | 0.639 (0.037) | 0.722 (0.018) | 1.000 (0.000) | 277.3 |
  | v6 C 0.03 | 0.517 (0.068) | 0.656 (0.040) | 0.708 (0.020) | 0.997 (0.016) | 292.5 |
  | v6 C 0.01 | 0.497 (0.088) | 0.628 (0.054) | 0.689 (0.011) | 0.970 (0.040) | 297.2 |
  | v6 without domain, C 0.1 | 0.533 (0.049) | 0.636 (0.056) | 0.692 (0.022) | 1.000 (0.000) | 272.4 |

  - No variant is better than C 0.1 by more than the std. Thus the record part stays v6 with C 0.1.
  - The llr of known and test agree: test full owners -2.47 (known full non-members -2.22), test victims 1.36 (known members 1.42). Thus no calibration shift between known and test occurs.
  - Scenarios with C 0.1 (mixture, prior shift w 1): victims with 75% members 0.927, 0.757. Clean 4% 0.776, 0.775. Hidden 3 1.000, 0.867. Hidden 12 1.000, 0.867. All wrong 0.718, 0.691.
  - Global simulation with C 0.1: 0.767 (std 0.014) against 0.756 (0.019). Photographer TPR at the global threshold 0.981 against 0.647.
- The coordinator accepted the mixture as the owner part of the photographers.
- Froze `src/photographer.py`:
  - The main function is `set_logodds(data, source_type)`. It works for all set types: C 0.1 for photographers, C 1 for the other types.
  - Removed the dead variants: tau, the owner mean, the record variants and the llr shift.
  - The honest check compares four variants: record, group prior, mixture, prior shift w 1.
  - The global simulation also runs with the expected test members of the mixture for all set types, not only with the members of attacks-v8a.
- Added the job `2026-11-27_53_photographer-v4` (CPU). It writes `submission_photographer_logodds.csv` (`id,logodds`, all 1440 photographer records) to the output.
- Job `photographer-v4` is complete in 2 minutes. Final results of the photographer module (v6 with C 0.1, owner mixture):
  - Known CV (validation of `src/validation.py`, baseline folds): record 0.531 (std 0.062), AUC 0.902. With the prior of the owner group: 0.639 (0.037), AUC 0.952. The baseline attacks-v8a has 0.503 (0.066), AUC 0.894.
  - Honest check, base scenario: mixture 1.000 (0.000), prior shift w 1 0.867 (0.026), record 0.722 (0.018).
  - Global simulation with the weights of attacks-v8a: 0.767 (0.014) against 0.756 (0.019). With the expected members of the mixture for all set types: 0.807 (0.015) against 0.753 (0.020).
  - Test: 277.3 expected photographer members, 292 records with a log-odds above 0. The structure of the text types gives 280.
- The output `submission_photographer_logodds.csv` of the kernel `nataliachoszczyk/bumas-p3-photographer-v4` has `id,logodds` for all 1440 photographer records (600 known out of fold, 840 test). Git ignores it, because the repository is public.
- Merged the branch `pro3-photographer` into `pro3-experiments`. The photographer module (`src/photographer.py`) is an owner mixture: each owner is a victim or clean, and EM on the test records estimates the probability.
- Added `explore_v10` to `src/attacks.py`: the owner mixture for all set types, the record score for forum posts and web photos.
- Job `attacks-v10`: on test, the mixture of the text set types is almost exact (r = 0.99 to 1.00, eps = 0). The global simulation on known gives 0.812 (std 0.015).
- The expected metric from the calibrated test scores is 0.885. For attacks-v8a the same estimate is 0.804, and the public score is 0.774. Thus the estimate is approximately 0.03 too high.
- Now the set types are almost exact. Forum posts and web photos use most of the false positive budget, so they are the next targets.
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
- Jobs `webphoto-matrix` (43 minutes) and `webphoto-grad` (13 minutes) are complete for all photos.
- Job `webphoto-v2` results inside the web photos (TPR@5%FPR, mean and std):
  - The rank features (own photo against 31 other photos, own caption against 31 other captions), the flipped photo and the Renyi entropies give no gain. Each alone has an AUC of 0.49 to 0.56.
  - The probes on the hidden states give AUC 0.51 to 0.55 alone and no gain in the stack. This agrees with the papers above.
  - The gradient norms have AUC 0.57 alone (members have a smaller gradient), but they follow the loss. Base + gradients with C = 0.01 gives 0.144 (0.016), AUC 0.637.
  - Global simulation (owner prior w = 0 / w = 1): v8a 0.741 / 0.756, base with C = 0.01 0.753 / 0.761. The web photo false positives at the global threshold fall from 27.3 to 15.9 (w = 1). Thus the calibration of the web photo log-odds is the main gain.
- Job `webphoto-v3` (CPU): the final choice of C and of the features, with the paired gain over v8a in the simulation.
- Job `webphoto-v3` is complete. A stronger or a weaker regularization, the gradient norms and the small feature sets give no gain larger than the std. In the global simulation, base C = 0.01 has the best paired gain over v8a: +0.012 (0.009) at w = 0 and +0.005 (0.007) at w = 1.
- Final choice for the web photos: the v8a photo features, one logistic regression with C = 0.01 (`logodds_all(photo_data(records), lr_score(BASE, 0.01))`). It needs only the passes of the earlier jobs. The log-odds of all web photos are in the output `webphoto-v3/logodds_baseC0.01.csv` of the kernel `jankombajn/bumas-p3-webphoto-v3`.

## 2026-11-27 (coordinator, branch `pro3-experiments`, after the modules)

- Job `shadow-b` is complete. Job `attacks-v8b` uses both shadow pairs: forum posts 0.855 (std 0.023, v8a: 0.843), all known 0.643.
- Merged the branch `pro3-webphoto`. The web photo module keeps the v8a features with C = 0.01. No new feature gives a gain larger than the std.
- Job `attacks-v10b`: both shadow pairs, the owner mixture for all set types, C = 0.01 for web photos. Global simulation: 0.813 (std 0.015).
- Added a test-like simulation: it drops the known non-members of victims, because in test a victim has only members.
- Job `attacks-v10c`: the test-like simulation gives 0.867 (std 0.018). This is the current best estimate of the test metric.


## 2026-11-27 (branch `pro3-forum`)

The branch `pro3-forum` comes from `pro3-experiments` (after `attacks-v10`). It holds the work on the forum posts. The Kaggle account is `nataliachoszczyk`, the job numbers are 54 to 69.

Baseline: the v6 record score, CV TPR@5%FPR 0.843 (std 0.027), AUC 0.953. The best single feature is `lira_gain` (AUC 0.93).

### Research summary

- Carlini et al. 2022 (LiRA): the attack becomes stronger with more shadow models. Online LiRA uses IN and OUT models, offline LiRA only OUT models. A variance for each record (or token) from the shadow models is important.
- Zarifzadeh et al. 2024 (RMIA): it compares the ratio target/reference of a record with the same ratio of population records. With one reference model it stays strong. In a logistic regression on the gain, the rank against the population adds little, because it is a monotone change of the same ratio.
- Hayes et al. 2025 (strong MIA on LLMs): LiRA with many reference models works on LLMs, but the gain for one record stays limited.
- Xie et al. 2024 (ReCaLL): the drop of the likelihood with a non-member prefix is larger for members. It is strong on WikiMIA, but weaker on MIMIR, where no distribution shift occurs.
- Zhang et al. 2025 (Min-K%++): already in the features.

Ideas, ranked by the expected value and the cost:

1. More shadow models for forum posts (online and offline LiRA with a variance for each token). Each model trains on one random half of the forum posts and 5% of the other texts, and scores only the forum posts. Cost: approximately 2 GPU hours for 8 models.
2. A calibration of the gain for the length and the difficulty of a post (for example the sum of the gain instead of the mean, and a z-score). Cost: CPU only.
3. ReCaLL with a prefix of known non-member forum posts, as a feature with the OUT shadow models for calibration. Cost: approximately 1 GPU hour.
4. An RMIA-style rank against population records. Cost: CPU only, small expected gain.

### Jobs

- Added `src/forum.py`, the one code file of the forum module. It uses `train` and `shadow_halves` of `src/refmodel.py` and changes no shared file.
- Added the job `2026-11-27_54_qwen-copy` (CPU). It downloads Qwen3-0.6B-Base with the revision of job 05 into an output of the account `nataliachoszczyk`, because the GPU jobs of this account cannot read the kernels of `jeronimoo`.
- Added the job `2026-11-27_55_forum-shadow-smoke` (GPU T4 x2): `run_forum_shadow` on 40 forum posts and 10 other texts, then a score of 20 forum posts.
- Added to `src/forum.py` the length and difficulty features of the forum posts (`token_features`):
  - the sum and the t statistic of the token gain over the OUT shadow model and over Qwen3-0.6B-Base,
  - the log of the length and the mean log p of the OUT model (the difficulty),
  - the mean gain on hard and on easy tokens.
- Added the forum CV with the baseline folds (`forum_cv`) and the log-odds file of the forum posts (`forum_logodds`).
- Added the job `2026-11-27_56_forum-v1` (CPU): v6, v6 with C 0.1, v6 with the length features, both changes together.
- Job `qwen-copy` is complete in 2 minutes.
- Job `forum-shadow-smoke` is complete in 6 minutes, without an error. To load the weights takes approximately 3 minutes. Two optimizer steps take 47 seconds.
- Job `forum-v1` is complete in 1 minute. TPR@5%FPR (mean, std): v6 0.843 (0.027), v6 C 0.1 0.843 (0.027), v6 + length 0.856 (0.024), v6 + length C 0.1 0.858 (0.029).
  - The gain of the length features is smaller than the std. Thus no decision yet.
  - Alone, the t statistic of the gain (`g_t`) has AUC 0.929, the same as `lira_gain`. The sum of the gain (`g_sum`) is weaker (AUC 0.902).
- Added to `explore` the paired difference against the first variant (the same seeds and folds). It is more sensitive than two separate stds.
- Added the job `2026-11-27_57_forum-shadow` (GPU T4 x2): 4 half splits of the forum posts, 8 models. Each forum post gets 4 IN and 4 OUT models.
- Added ReCaLL to `src/forum.py` (`run_recall`). It computes log p of each forum post after a prefix of 4 known non-member forum posts. BUMAS-mini gives the score, and Qwen3-0.6B-Base gives the same value for the calibration.
- Added the jobs `2026-11-27_58_recall-smoke` (20 posts) and `2026-11-27_59_recall` (all 3000 forum posts), GPU T4 x2.
- Job `recall-smoke` is complete in 10 minutes, without an error. Started the job `recall` in the second GPU session.
- Added `recall_features` and the variants of the job `forum-v2` to `src/forum.py`.
- Added the job `2026-11-27_60_forum-feature-smoke` (CPU). It checks `forum_shadow_features` and `recall_features` on the outputs of the two smoke jobs, before the long jobs end.
- Job `forum-feature-smoke`, version 1 failed. `forum_shadow_features` is correct (20 records). `recall_features` failed when no post of the prefix was in the table. Changed the median fill to one column at a time. Version 2 runs the same check.
- Job `forum-feature-smoke`, version 2 is complete. Both feature functions give values for the 20 smoke posts.
- Job `recall` is complete in 75 minutes (1.2 seconds for each post and model).
- Added the job `2026-11-27_61_forum-v2a` (CPU): v6, v6 + ReCaLL, v6 + length. The forum shadow variants follow when the job `forum-shadow` ends.
- Job `forum-v2a` is complete in 1 minute. TPR@5%FPR (mean, std; paired difference against v6, mean and std):
  - v6: 0.843 (0.027).
  - v6 + ReCaLL: 0.849 (0.023); +0.006 (0.012).
  - v6 + length: 0.856 (0.024); +0.013 (0.011).
  - The ReCaLL features have no signal alone (AUC 0.46 to 0.52). The members and the non-members come from the same distribution, as in MIMIR. Thus ReCaLL is removed.
  - The gain of the length features is approximately one std of the paired difference. They stay a candidate.
- Job `forum-shadow` is complete in 2 hours 35 minutes, no error. All 8 models scored the 3000 forum posts.
- Added the job `2026-11-27_62_forum-v2` (CPU): all variants of `V2`, with the forum shadow features.
- Job `forum-v2` is complete in 1 minute. TPR@5%FPR (mean, std; paired difference against v6, mean and std):
  - v6: 0.843 (0.027).
  - v6 + forum shadow: 0.858 (0.029); +0.015 (0.014).
  - v6 + ReCaLL: 0.849 (0.023); +0.006 (0.012).
  - v6 + length: 0.856 (0.024); +0.013 (0.011).
  - v6 + forum shadow + length: 0.872 (0.019); +0.028 (0.014). This gain is two stds of the paired difference.
  - Single features: the gain over the mean of the 4 OUT forum shadow models (`f_gain`) has TPR@5%FPR 0.833 and AUC 0.948. The gain over one OUT model of the full shadow pair (`lira_gain`) has 0.739 and 0.929. The offline z-score (`f_offline`) has 0.830 and 0.950. The online LiRA ratio is weaker (0.509, AUC 0.853).
- More OUT models make the gain feature much stronger. Thus added the job `2026-11-27_63_forum-shadow-b` (GPU T4 x2) with 4 more splits (4 to 7). With both jobs, each forum post has 8 IN and 8 OUT models.
- Changed `forum_shadow_features`: it reads the splits of all job outputs below `/kaggle/input`, and `max_splits` can limit them.
- Merged `origin/pro3-experiments` (the web photo module, `explore_v10` with `single_C`, the test-like global simulation).
- Added the job `2026-11-27_64_forum-v3` (CPU), on the 4 splits of `forum-shadow`:
  - v6 + forum shadow + length (the best model of `forum-v2`), with C 1 and C 0.1,
  - only the three strong forum shadow features (`f_gain`, `f_gain_t`, `f_offline`),
  - a compact model: 5 v6 features, the forum shadow and the length features,
  - the global simulation of `explore_v10` (shadow pair `shadow`, web photo C 0.01) without and with the forum file.
