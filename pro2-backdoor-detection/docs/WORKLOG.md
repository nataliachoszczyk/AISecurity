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
