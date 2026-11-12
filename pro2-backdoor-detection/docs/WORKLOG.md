# Worklog - Project 2

## 2026-11-09

- Made the project folder with `docs/`, `jobs/` and `src/`.
- Added the task text in Polish (`docs/task/task.md`) and in English (`docs/task/task_eng.md`).
- Changed `jobs/kaggle.sh`: it uses `.kaggle/access_token` and `.venv` from the root folder of the repository.
- The Kaggle account of the project is `jeronimoo`.

## 2026-11-10

- Decision: the task data does not go to the local machine. A Kaggle CPU job gets it from Google Drive.
- Added the job `2026-11-10_01_data`. Its output becomes the private dataset of the project.
- Checked the file list of the Drive folder: 610 files (200 test models, 6 train models, 400 clean images, 4 other files).
- The job `data` failed at first, because the account `jeronimoo` had no phone verification and the kernel had no internet.
- Version 2 of the job `data` is complete: all 610 files (4.62 GB) are in the kernel output.
- The private dataset `jeronimoo/bumas-p2-backdoors` has the output of the job `data`: 610 files, 4.62 GB. The account `jankombajn` is a reader of it.
- Decision: `model.py` of the task stays in the dataset. The repository has no copy of it.
- Added `src/paths.py`. It finds the task data and the output folder on Kaggle and on the local machine. It also loads a model with `model.py` from the dataset.
- The folder `example/` is also in the local `data/` folder (not in git). It shows the train labels: three models with one backdoor each (ASR 0.964, 0.999 and 0.581) and three clean models. The clean accuracy is 0.84 to 0.85 for all six models.

## 2026-11-11

- The work continues on the branch `pro2-experiments`. The Kaggle jobs run on the account `jeronimoo`.
- Decision: no Neural Cleanse baseline. The models have the same base, the same data and the same finetune. Thus a comparison of a class with the same class in the other models can show a backdoor.
- Added `src/model_stats.py`. It collects statistics of the weights and of the answers of each model. It compares each class with the same class in all 206 models (z-score). The seed is 0.
- Added the job `2026-11-11_02_model-eda`. It is the EDA of the models.
- Version 1 of the job `model-eda` failed: the image id in `clean/labels.csv` has no file extension. The code now searches for `images/<id>.*`.
- Version 2 of the job `model-eda` failed at the AUC step: a z-score of a statistic with std 0 was NaN. Now such a z-score is 0. The job saves the raw tables first and catches errors of the analysis, because Kaggle deletes all output of a failed job.
- Version 3 of the job `model-eda` is complete (9 minutes on a T4). The small outputs are in `jobs/runs/2026-11-11_02_model-eda/output/model-eda/`.
- Result: the finetune changes all layers by approximately 5% to 25% of their norm. The poisoned train models have the same layer distances as the clean train models.
- Result: the output layer of each model is very different from the mean (relative distance 1.8), because the finetune starts it from a random seed.
- Result: the weight sketches (PCA) show no groups of models. The poisoned train models are not separate from the clean train models.
- Result: the best statistic is `z_clean_logit`: the mean logit of a class on the clean images of the other classes, compared with the same class in all models. On the train pairs, its AUC is 0.96, and the target classes have the ranks 1, 1 and 7 in their models.
- Result: Gaussian noise activates the backdoor of one train model (`z_gauss_logit` = 11). The fraction of answers with high confidence on random noise changes much between models (0.32 to 1.00), also for clean models.
- Note: the train set has only 3 positive pairs, and the job tested 34 statistics. Thus the AUC values are not reliable.
- Check of the train score: the AUC of `z_clean_logit` is 0.957 (bootstrap over the 6 models, 5% to 95%: 0.83 to 1.00).
- Check of the train score: the best of 34 random statistics gets a median AUC of 0.88 on the same 120 train pairs (95th percentile 0.95). Thus the 6 train models cannot show the difference between methods.
- Check of the train score: leave-one-model-out with the selection of the best statistic gives an AUC of 0.33.
- Decision: the project needs its own validation set of models with known backdoors.
- Added `src/val_models.py`. It makes validation models with known backdoors. It finetunes the 3 clean train models for 30 epochs on 200 of the 400 clean images. The other 200 images (holdout) measure the ASR and the clean accuracy. A detector must use only the holdout images on these models.
- The plan has 120 models: 60 clean, 35 with 1, 17 with 2 and 8 with 3 backdoors. The trigger types are patch, blend, sig, warp, noise and color. SIG is clean-label. 20% of the other backdoors are targeted on one source class. The clean models get the same finetune without poison.
- A model is valid only if each backdoor has an ASR of 0.5 or more. After a failure, the next attempt makes the triggers stronger (at most 4 attempts). The seed is 0.
- Added the jobs `2026-11-11_03_val-models-smoke` (12 models) and `2026-11-11_04_val-models` (120 models).
- Version 1 of the job `val-models-smoke`: the finetune decreased the clean accuracy to 0.60, also for the clean models. The real train models have 0.80 to 0.83. The warp trigger got an ASR of 0.10 to 0.36 after 4 attempts. 2 of 4 poisoned models were valid. One finetune took approximately 17 seconds.
- Changed the finetune: the BatchNorm layers stay in eval mode, the learning rate is 0.002 (not 0.005), and a distillation loss keeps the answers of the base model on the clean finetune images. The warp amplitude starts at 2 to 3 pixels (not 1 to 2).
- Version 2 of the job `val-models-smoke`: all models, also the clean models, gave one class for almost all images (clean accuracy 0.05). The cause is not known.
- Version 3 of the job `val-models-smoke` is a diagnosis. It compares 3 finetune settings on 3 models: BN in train mode with distillation, BN in eval mode with distillation, BN in eval mode without distillation. It writes the loss and the base accuracy.
- Version 3 of the job `val-models-smoke`: the base models have an accuracy of 0.81 on the holdout images, so the measurement is correct. A finetune on 200 images decreases the accuracy in all settings (BN train with distillation: 0.62 to 0.69; BN eval without distillation: 0.64 to 0.69; BN eval with distillation diverges to 0.05). Without distillation, the train loss goes to 0.003: the model learns the 200 images by heart. Only 1 of 6 backdoors got an ASR of 0.5 or more.
- Conclusion: 200 images are not sufficient for a realistic finetune. An option is the Kaggle dataset `ayaroshevskiy/downsampled-imagenet-64x64` (ImageNet at 64 x 64, all 1000 classes). The 20 task classes are ImageNet classes.
- Decision: the validation models use ImageNet 64x64 images of the 20 task classes. The dataset `ayaroshevskiy/downsampled-imagenet-64x64` has no labels, so the jobs use `j53t3r/imagenet64x64` (the format of Chrabaszcz et al., with labels). Its label order is not the usual ImageNet order.
- Changed `src/val_models.py`:
  - The job finds the ImageNet64 label of each task class. The 3 clean base models classify the 50 000 ImageNet val images, and a linear assignment selects the label with the most votes for each class (`class_map.csv`).
  - The finetune set has approximately 1 300 ImageNet train images for each class (approximately 26 000 images). The ImageNet val images (50 for each class) measure the ASR. The 400 task images measure the clean accuracy and stay free for the detectors.
  - The finetune is like the finetune of the task: a new random output layer, then 4 epochs on all layers (SGD, one-cycle learning rate 0.01, batch 128, crop and flip, mixed precision).
  - A clean-label or targeted poison takes at most half of one class.
  - The job uses the 2 GPUs of a T4 x2 session (one process for each GPU). It stops a GPU process if the first model has a clean accuracy below 0.7, and it stops new models after 10.5 hours.
- Added `imagenet_dir()` to `src/paths.py`.
- Version 4 of the job `val-models-smoke` failed: the import of `src` found no module. `run.py` adds the folder to `sys.path` before the clone, so the import cache must be cleared after the clone. Added `importlib.invalidate_caches()`.
- Version 5 of the job `val-models-smoke` failed at the save of the cache: the folder `/kaggle/temp` did not exist. The class map was clear for most classes (score 0.69 to 0.98), but for 4 classes the second label had almost the same score (for example toucan 0.89 and 0.87).
- The dataset `j53t3r/imagenet64x64` has two copies of each file. The code now keeps one path for each file name (51 740 images became approximately 25 900).
- Added a check of the class map: the fraction of task images with a near copy in the ImageNet train images of the mapped label.
- Version 6 of the job `val-models-smoke` is complete (16 minutes). The finetune set has 25 870 images (approximately 1 300 for each class). 4 models: ASR 0.80 to 0.99, ImageNet val accuracy 0.91 to 0.93. The clean model has an accuracy of 0.80 on the task images (base 0.81). One model takes approximately 150 seconds.
- No task image has a near copy in ImageNet64, so the copy check cannot confirm the class map. The task images probably have a different downsampling.
- Problem: the warp model has an accuracy of 0.615 on the task images, but 0.91 on the ImageNet val images. Probably the model learned the blur of the warp, and the softer task images partly activate the backdoor.
- Added the noise mode of WaNet: for each warp backdoor, the same number of other images gets a different random warp and keeps its label. The early accuracy check now uses only clean models.

## 2026-11-12

- Version 7 of the job `val-models-smoke` is complete (15 minutes). All 5 models are valid. ASR: 0.63 to 0.99. Clean accuracy on the task images: 0.735 to 0.82 (warp 0.74 after the noise mode, clean model 0.80, base 0.81). A targeted patch backdoor got an ASR of 0.98.
- Started the job `2026-11-11_04_val-models` (120 models, T4 x2).
- Check of the class map with `map_clsloc.txt` of the ILSVRC devkit (the label order of ImageNet64): 18 of 20 classes were correct. The vote of the base models selected Welsh_springer_spaniel (ID 19) for Walker_hound (ID 18), and photocopier (ID 789) for ashcan (ID 752). The 4 uncertain classes of the vote (toucan, dugong, Walker_hound, yawl) were correct except Walker_hound.
- The code now uses the fixed list `IMAGENET64_LABELS` from `map_clsloc.txt`. The vote of the base models stays as a check (`class_map.csv`).
- Version 1 of the job `val-models` used the wrong images for these 2 classes. It must be cancelled and started again.
- The user cancelled version 1 of the job `val-models` after 55 minutes. Started version 2 with the fixed class map.
- Decision (with the user): a second, sealed validation set B. Its trigger types are not in set A. A detector sees set B only once, before a submission. The rule of the user: a submission only if the detector gets an AUC of 0.96 or more on our independent validation models.
- Set B has 80 models: 40 clean, 24 with 1, 11 with 2 and 5 with 3 backdoors. The trigger types come from the attacks of BackdoorBench that set A does not have:
  - `reflect` (ReFool): a blurred ImageNet image of another class, added like a reflection.
  - `quantize` (BPP): fewer levels for each color channel.
  - `lowfreq` (LF): a smooth additive pattern.
  - `ftrojan` (FTrojan): an additive pattern at a high frequency.
  - `dynamic` (like SSBA): a change that depends on each image, through a fixed random filter.
  - `patch` and `blend`, only as clean-label.
- In set B, 15% of the backdoors of the new types are clean-label and 20% are targeted. The seeds of set B start at 1000.
- `plan_models` and `run` have a set name ("a" or "b"). Set A gives the same plans as before.
- Added `src/detect.py`. For each pair of classes (source, target), it optimizes a small universal change on 4 images of the source class and measures it on the other 16 images. A backdoor gives a change that works on all images of the class. There are 4 families of changes: `additive` (L-infinity 0.03), `patch` (mask and pattern, with a penalty on the mask size), `color` (a 3 x 3 color matrix and a bias), `warp` (a smooth flow of at most 2 pixels). It also keeps the population statistics of `src/model_stats.py`.
- Added `src/evaluate.py`. It compares each value with the same value (same pair or same class) in all models of the same set (robust z-score with the median and the MAD). It gives the AUC of the task on the labeled models, also for each trigger type and variant. The report does not use the sealed set B.
- Added the jobs `2026-11-12_07_detect-smoke` and `2026-11-12_08_detect-v1`.
- Version 1 of the job `val-models-b-smoke` is complete (27 minutes). 7 of 8 models are valid. All new trigger types work (ASR 0.52 to 0.99). A clean-label patch is the most difficult: one model failed after 3 attempts (ASR 0.275). The clean model has an accuracy of 0.88 on the task images (0.80 in set A before the fix of the class map). This also confirms the fix.
- Version 1 of the job `detect-smoke` is complete. The code works from the features to the report. One model takes 250 seconds, so 400 models need more than 28 GPU hours.
- Changed the detector for speed: 25 steps (not 40), 3 optimization images for each class (not 4), channels_last, cudnn benchmark. The detection runs in one job for each set (`detect-v1-a`, `detect-v1-real`, `detect-v1-b`), because the z-scores use only the models of one set.
- Added the combiner to `src/evaluate.py`: a logistic regression on all scores, with a cross-validation that leaves one trigger type out. The job `detect-v1-a` reports it.
- Started the job `val-models-b` (80 models of the sealed set B).
- The jobs `val-models` (set A) and `val-models-b` (set B) are complete.
  - Set A: 114 of 120 models are valid: 60 clean, 34 with 1, 13 with 2, 7 with 3 backdoors. Mean ASR: patch 1.00, noise 0.98, blend 0.96, warp 0.92, color 0.87, SIG 0.72. Mean clean accuracy on the task images: 0.86 (clean models), 0.82 (1 backdoor), 0.78 (2 or 3 backdoors).
  - Set B: 69 of 80 models are valid: 40 clean, 21 with 1, 6 with 2, 2 with 3 backdoors. Mean ASR: ftrojan 0.99, quantize 0.93, dynamic 0.88, reflect 0.80, blend (clean-label) 0.80, lowfreq 0.76, patch (clean-label) 0.54.
  - The clean accuracy falls with more backdoors. A detector must not use only the accuracy. The evaluation checks this with the score `stat_clean_recall`.
- Started the jobs `detect-v1-a` and `detect-v1-real`.
- The user made the private datasets `jeronimoo/bumas-p2-val-models-dataset` and `jeronimoo/bumas-p2-val-models-b-dataset` from the outputs of the jobs. The next versions of the detector jobs use these datasets, not the kernel outputs. Version 1 of `detect-v1-a` still uses the kernel output.
- Version 1 of the job `detect-v1-a` is complete (2.4 hours, 150 s for each model, not 85 s).
  - Best single scores: `warp_max` 0.938, `inv_max_max` 0.924, `additive_max` 0.920. For 5 of 6 trigger types, the best scores get an AUC of 0.96 to 1.00.
  - SIG is the problem: AUC 0.53 or less, often below 0.5. The SIG change has an amplitude of 0.12 to 0.15. The additive family allows only 0.03, so it cannot make the stripes.
  - The clean accuracy (`stat_clean_recall`) gives an AUC of 0.50. Thus the accuracy alone does not show the backdoors.
  - The combiner with leave-one-type-out gets only 0.82 (color 0.63, SIG 0.15). It overfits to the trigger types of the train folds.
- Detector v2: the new family `smooth` (an additive change of at most 0.1 on a 32 x 32 grid, then a bilinear upsample). New scores: the mean of the 3 largest z-scores, the second largest family, and the mean rank of the families. A compact combiner with 6 inputs.
- Version 1 of the job `score-v1-a` is complete. The best score is `inv_top3_2nd` (the second largest family of the top 3 mean): AUC 0.940. It gets 0.97 to 1.00 for all types except SIG (0.44). The compact combiner gets 0.874 with leave-one-type-out, better than all inputs (0.789), but worse than the best single score.
- SIG decides the result now: 8 of 81 positives, and all scores rank them below the clean classes. `stat_clean_logit` has an AUC of 0.16 for SIG: a SIG model gives a lower logit to the clean images of the target class. A probable reason: the model uses the stripes as a feature of the target class, and the clean images do not have them.
- Added the scores `mix_low_*` and `mix_abs_*`: `inv_top3_2nd` plus a low (or unusual) logit of the clean images of the class. This is a test of the opposite sign of SIG. Set B has clean-label patch and blend, so it can show if this transfers. Version 2 of `score-v1-a` reports it on the features of `detect-v1-a`.
- Version 2 of `score-v1-a`: the mix with the clean confidence helps. `mix_abs_1.0` gets an AUC of 0.966, `mix_low_1.0` 0.965 (SIG 0.70 and 0.79, the other types 0.93 to 1.00). The weight was selected on set A with the SIG labels, so this is optimistic. Set B must confirm it.
- Rule of the user (2026-11-12): no new Kaggle job without the approval of the user. The user approved the 2 CPU jobs `score-v1-real` and `score-v2-a`.
- Added `submission()` to `src/evaluate.py`. The submission files stay in `submissions/` folders, which git ignores.
