# Worklog - Project 1

## 2026-10-31

- Made the project folder and the shared files of the repository (`.gitignore`, `requirements.txt`, main README).
- Added the task text in Polish (`task.md`) and in English (`task_eng.md`).
- Added the EDA: `src/eda.py` (logic) and `notebooks/01_eda.ipynb` (results). Added `scipy` to `requirements.txt`.
- Train set: all patches and all decoys are in the bottom half of the image (y_min >= 32).
- Test set: a 3x3 median on one half of the image changes the prediction. In the test set, the top half causes the change in approximately the same number of images as the bottom half. In the train set, the top half almost never causes it. Thus approximately half of the test patches are probably in the top half.
- The total strength of the attacks in the test set is approximately the same as in the train set.
- Added the consequences for training to the EDA conclusions: flip augmentation during training (vertical and horizontal, p = 0.5, fixed seed), box change with the flip, no color or noise augmentation.

## 2026-10-31 (evaluation and first method)

- Added `src/config.py` (seed, val fraction, experiment configs in `configs/*.yaml`) and `pyyaml` to `requirements.txt`.
- Added `src/evaluate.py`: the score function of the server, a report for each image kind and attack, the train/val split and the evaluation sets.
- Made the split once: `splits/split.csv`, 4500 train and 500 val, stratified by image kind and attack, seed 0.
- Checked val_flip: after a vertical flip, smoothing of the patch gives the true class for 7% of the patch images (93% without the flip). ResNet-18 misclassifies upside-down images, so val_flip does not show the work of a method on patches in the top half.
- Added `src/saliency.py`: integrated gradients toward a smoothed image give candidate boxes, and a causal check with ResNet-18 selects the patch. The method needs no training.
- Added `jobs/` with `JOBS.md` (job list and status) and the job scripts. The smoke test found that Kaggle runs Python 3.13.
- Ran the job `saliency` on Kaggle (2x T4, 22 min). Val: saliency-v1 (median 3x3) 0.530, saliency-v2 (Gaussian blur) 0.525. Clean 0.88, decoy-only 0.94, patch images 0.30-0.51 (MI-FGSM is the worst).
- The weak part is the box: the best of the 32 candidates has a mean IoU of only 0.58 on patch images. Maps with only the positive part of the gradients were worse (0.35).
- Made `submissions/submission_saliency-v1.csv` for the leaderboard: 20.9% of the answers are 0,0,0,0, and 51% of the boxes are in the top half.
- Added `src/unet.py`: a UNet on the image and its noise residual predicts the masks of the patch and the decoys. The causal check with ResNet-18 selects the patch. Flips and 90 degree rotations are the augmentation, and the prediction is the mean over all 8.
- unet-v1 (image and residual only): val 0.350, mask IoU 0.21. The UNet does not find the weak noise well from 4000 images.
- unet-v2 (with the integrated gradients map as input): val 0.508, mask IoU 0.33. Detection is almost perfect (clean 1.00, decoy-only 0.98), but the boxes are worse than in saliency-v1 (0.530).
- Changed `jobs/wait.sh`: it writes each status change of a job into `JOBS.md`.
- Added the small outputs of each Kaggle run to `jobs/<job folder>/<codename>/`: `metrics.json`, `report_*.csv` and the train loss. Submission files stay out of git, because the repository is public.
- `src/unet.py` now writes the loss of each epoch to `train_log.csv`.

### Conclusions on 2026-10-31

- The best method is saliency-v1 (val 0.530). The file `submission_saliency-v1.csv` is ready, but nobody sent it to the leaderboard yet. Git does not keep it. Download it from the output of the Kaggle notebook `bumas-p1-saliency`, or run the job again.
- Detection (patch or no patch) is good in all methods. The unet-v2 detection is almost perfect (clean 1.00, decoy-only 0.98).
- The box is the weak part. MI-FGSM patches get the lowest points in all methods (0.21-0.32).
- The UNet loss still decreased at epoch 60 (`jobs/2026-10-31_0[34]_unet-v[12]/unet-v*/train_log.txt`). Thus the training was too short.
- val_flip is not valid (see above). At this time we have no valid check for patches in the top half. The UNet is equal for top and bottom by design (8 flips and rotations in training and prediction). The saliency method has no training.

### Next steps

1. unet-v3: the same as unet-v2, but with more epochs (for example 200) and a check of the loss on the tune part.
2. Combine the methods: detection from unet-v2, box from the best of the saliency and UNet candidates.
3. Generate attacks with ResNet-18 (PGD, MI-FGSM, C&W, with a mask and JPEG) on clean and decoy-only images. Use bases from the train part for training data. Use bases from the val part for a valid val set with patches in the top half and in the middle.
4. Decide if the submission file must also go out of the git history (commit 9776e14). This needs a force push.

## 2026-10-31 (unet-v3)

- Renamed the configs to `configs/YYYY-MM-DD_NN_<codename>.yaml`. The date and NN are the same as in the name of the job folder, so the files show the order of the runs. `load_config` finds the file from the codename.
- Split the job folder `2026-10-31_03_unet` into `2026-10-31_03_unet-v1` and `2026-10-31_04_unet-v2`. The two runs were different versions of one Kaggle notebook, but they are different experiments. Each experiment now has its own number.
- `src/unet.py` now calculates the loss on the tune part after each epoch. `train_log.csv` has the new column `tune_loss`. If the tune loss increases while the train loss decreases, the UNet overfits.
- Added unet-v3: the same as unet-v2, but with 200 epochs (unet-v2 had 60).
- Added the job `2026-10-31_05_unet-v3`. From now on, the jobs run on the Kaggle account `jankombajn`. The script stops at once if the kernel has no GPU.
- Sent saliency-v1 to the leaderboard: 0.5373 (val 0.530). Thus the val set gives a good estimate of the leaderboard score. The best team had 0.6008 at this time.
- A check of the noise residual: the mean residual energy inside a patch is approximately 1.1 times the energy around it. The decoys have the same ratio. Thus the noise alone does not show the box. The box must come from ResNet-18.
- Added `src/generate.py`: it makes new attacked images with the patch at any position. The bases are the clean and decoy-only images. For each base, it adds 0-2 new decoys, attacks one random box and saves the image as JPEG (quality 85-95). A binary search finds the smallest strength that works after JPEG. The attacks do not calculate the gradient through JPEG.
- The generated train set uses only bases from the train part, with flips and rotations. The generated val set uses only bases from the val part, without flips. Thus the generated val set is the first valid check for patches in the top half.
- Added the job `2026-10-31_06_gen-v1` (config `2026-10-31_06_gen-v1.yaml`): 16 images for each train base and 8 for each val base. The job also scores saliency-v1 on the generated val set and draws `examples.png`. The images stay on Kaggle: the next jobs use the output of this notebook as input.
- unet-v3 (200 epochs): val 0.524 (unet-v2 0.508). The tune loss was lowest at epoch 59 (0.774) and then increased to approximately 1.2. The train loss decreased to 0.35. Thus the UNet overfits on 4000 images, and a longer training does not help.
- A small check of gen-v1 on the CPU: all rotated images changed the ResNet-18 class. Thus a rotation makes a different base, and the attack on it is not like the task data.
- gen-v2 (job `2026-10-31_07_gen-v2`) is the one large generated set for the Kaggle dataset. Changes from gen-v1:
  - No rotations. Only the horizontal flip, because ResNet-18 keeps the class of most flipped images.
  - The flipped image with the new decoys must have the class of the upright base after JPEG. Other images are rejected. The log shows the number of rejected images for each reason.
  - The train part stops after 9 hours, so the zip is complete before the Kaggle limit.
  - The output is one zip file (`gen-v2.zip`) for the dataset. Make the dataset one time on the notebook page: Output, then New Dataset.
- Added two options to `src/unet.py`:
  - `best_epoch`: keep the weights of the epoch with the lowest tune loss.
  - `boxes: fit`: the candidates are the rectangles of an allowed size (8-15% of the area, ratio 1:2 to 2:1) with the largest sum of (probability - `fit_offset`). Six come from the patch mask and six from the decoy mask. The old method uses a threshold and connected parts.
- Job `2026-10-31_08_unet-v4`: unet-v4a (old boxes) and unet-v4b (fitted boxes). Both train with the same seed for 80 epochs and keep the best epoch. Thus the difference of the scores comes only from the box method.
- Found a data leakage in gen-v2: it used the clean and decoy-only images of the UNet tune part as bases. Then the UNet learns these backgrounds, and the tune score is too high. The gen-v2 run was cancelled.
- Added `tune_ids()` to `src/evaluate.py`. The UNet and the generator use it, so both have the same tune part. It gives the same 450 images as the old UNet code (370 ADV, 43 clean, 37 decoy-only). Thus the old results stay comparable.
- gen-v3 (job `2026-10-31_09_gen-v3`) is gen-v2 without the 80 bases of the tune part: 820 train bases and 100 val bases. Make the Kaggle dataset from this notebook.
- `src/unet.py` can now train on the generated set (option `generated`). It also scores the generated val set, with a report for the top and the bottom half. It stops if a generated image uses a tune image as its base.
- Data leakage checks:
  - The UNet stops if a base of the generated train set is not in the train part without tune.
  - The UNet stops if a base of the generated val set is in the train part.
  - The 10000 task images (train, val and test) have no exact duplicates and no near duplicates (8x8 mean image, 16 levels).

## 2026-11-01

- gen-v3 printed no progress after 14 minutes, because its first batch has 2000 images. The live log works: `kaggle kernels logs -f <kernel id>`.
- The generator now prints the time of each attack in each batch.
- The first batch of gen-v3 (val, 2000 jobs) took approximately 22 minutes (0.66 s for each job). Thus a speed test job is not necessary.
- Results of this batch:
  - 1145 images. 36 jobs were rejected by the class check and 819 by the attack (no strength level works after JPEG).
  - The targeted PGD worked only in 100 of approximately 490 jobs. The target was a random class.
  - The strength of PGD and MI-FGSM is the same as in the task data (median 10/255).
  - For C&W, the strength column had the constant of the attack, not the largest pixel change.
  - saliency-v1 on the generated val set (patch images only): 0.437 in the bottom half and 0.418 in the top half. Thus saliency-v1 also works for patches in the top half.
- gen-v4 (job `2026-11-01_10_gen-v4`) corrects the two problems:
  - The target of the targeted PGD is a near class: the class at rank 2-6 of ResNet-18. The task data has a smaller eps for this attack than for the others, so the authors probably used a near class.
  - The strength of C&W is the largest pixel change before JPEG, as in the task labels.
- Added the option `max_bases` to the generator and the job `2026-11-01_11_gen-smoke`. It runs the full flow of gen-v4 on the Kaggle CPU with 4 bases for each part. The CPU has no session limit, so the test does not wait for a GPU.
- The generator now makes the jobs in rounds: one copy of each base, then the next copy. Before, it made all copies of one base before the next base. Then a stop at the time limit removed the last bases completely.
- The ResNet-18 of the task has no downsampling in the stem (3x3 conv, stride 1, no max-pool). Thus each attack step costs approximately 2 GFLOP for each image, and the GPU is the limit.
- Added fp16 (autocast) to the attack steps. The final check of each attack stays in fp32.
- Added the job `2026-11-01_12_gen-speed`: the time of one PGD attack with fp32 or fp16, on 1 GPU or on 2 GPUs.
- unet-v4 results (val):
  - unet-v4a (old boxes, best epoch 55): 0.448. The weights of the best epoch alone do not help.
  - unet-v4b (fitted boxes, best epoch 59): 0.613. This is the best result. The fitted boxes give +0.165 on the same UNet.
  - unet-v4b by attack: C&W 0.578, MI-FGSM 0.399, PGD targeted 0.602, PGD untargeted 0.515, clean 0.96, decoy-only 0.98.
  - The two runs had the same seed, but different best epochs. Thus the GPU training is not fully deterministic.
- The submission of unet-v4b has 27% of 0,0,0,0 answers (20% expected). 51% of the boxes are in the top half.
- unet-v4b on the leaderboard: 0.5590 (val 0.613). saliency-v1 had no such gap (val 0.530, leaderboard 0.5373).
- A comparison of the two test submissions:
  - In 20.0% of the test images, both methods answer 0,0,0,0. This is the share of images without a patch.
  - In 7.0%, only unet-v4b answers 0,0,0,0, and saliency-v1 gives a box. These are probably patch images where the UNet masks are weak.
- Made `submission_unet-v4b-saliency.csv`: unet-v4b, but the saliency-v1 box where only unet-v4b answers 0,0,0,0 (349 images, 20.0% of 0,0,0,0 answers).
- Added the option `ig_candidates` to `src/unet.py`: more candidate boxes from the integrated gradients map. Job `2026-11-01_13_unet-v5`: unet-v4b with 24 of these candidates.
- gen-speed version 1: fp32 on 1 GPU takes 3.9 ms for each image step (batch 512). Batch 1024 in fp32 is out of memory on one T4. Version 2 catches this error and also measures batch 256.
- unet-v4b-saliency on the leaderboard: 0.5648 (unet-v4b 0.5590). The difference is smaller than 0.02, thus it can be random.
- Decision: send a submission only after a clear improvement on val.
- Added `estimate_jpeg_quality()` to `src/eda.py`: the quality (50-96) with the smallest change after one more JPEG save. On 40 train images, 95% of the estimates are equal to the label, and all are within 1.
- Added the job `2026-11-01_14_jpeg-quality` (CPU): the estimate for all train and test images. It checks if the test images have a lower JPEG quality.
- gen-smoke passed the full flow on the CPU (generation, save, examples, saliency check, zip). It found one problem: the C&W strength was 17-20/255 (task data: median 9.6/255).
- C&W now uses an Adam step of 0.001 (before 0.01). With the larger step the change reached the 20/255 limit after a few steps, and the L2 term could not make it small.
- Added `check_strength()`: after the generated val set, the run stops if the median strength of an attack differs from the task data by more than 4/255. Thus a wrong attack does not use 9 hours of GPU.
- jpeg-quality results: the estimate is equal to the train label for 94.3% of the train images, and within 1 for 99.9%.
  - Train: quality 90-96 only.
  - Test: 995 images (approximately 20%) have a quality of 80-88. The other test images have 90-96. These are probably the "more difficult" test images of the task.
  - unet-v4b answers 0,0,0,0 for 30.0% of the test images with quality 80-88, and for 26.3% of the others. saliency-v1: 21.8% and 20.7%.
- gen-v4 now uses a JPEG quality of 80-95 (config `jpeg`). Thus the UNet learns patches after a stronger compression, and the generated val set can measure this case. The real val set has no such images.
- The reports of the generated val set now also show the JPEG quality bands 80-89 and 90-95.
- gen-speed: DataParallel on 2 GPUs is slower than 1 GPU (fp16: 1.73 ms against 0.93 ms for each image step). fp16 is approximately 3.6 times faster than fp32 (0.93 ms against 3.37 ms), with the same success after JPEG.
- The generator now runs one process on each GPU (`_generate_shard`), each with every second job. The two parts are joined at the end.
- unet-v5 (unet-v4b plus 24 candidates from the integrated gradients map): val 0.592 (unet-v4b 0.613). Clean images: 0.86 (unet-v4b 0.96). The extra candidates give boxes on clean images. Thus this idea does not help.
- unet-v4b and unet-v5 had different best epochs (59 and 55) with the same seed. The GPU training is not deterministic, so a difference of approximately 0.01-0.02 on val can be random.
- Prepared unet-v6 (job `2026-11-01_15_unet-v6`): unet-v4b on the real train images plus the generated train set, 40 epochs. It needs the Kaggle dataset `jankombajn/bumas-p1-gen` from the output of gen-v4.
- gen-v4 version 1 failed after 2.5 hours. The output of a failed Kaggle run is lost.
  - The generated val set was complete: 1100 images. saliency-v1 on it: 0.411 (JPEG 80-89: 0.383, JPEG 90-95: 0.448).
  - The check of the attack strength passed (medians 8-12/255, task data 8-10/255).
  - The train shard on GPU 1 was complete: 15514 images from 26240 jobs (59%).
  - The train shard on GPU 0 failed immediately with "CUDA out of memory". The main process kept 12 GB of cache on GPU 0 after the saliency check.
- `run_generation()` now empties the CUDA cache after the val checks. gen-v4 version 2 runs with this fix.
- unet-v6 now reads the output of the gen-v4 notebook (`kernel_sources`), not a Kaggle dataset.
- gen-v4 version 2 is complete (2 hours 20 minutes). The output `gen-v4.zip` has the generated train and val sets.
  - Train: 30436 images from 52480 jobs (58%). Medians of the strength: C&W 7.1, MI-FGSM 8, PGD targeted 10, PGD untargeted 8 (/255).
  - Val: 1100 images, the same as version 1. saliency-v1 on it: 0.411 (JPEG 80-89: 0.383, JPEG 90-95: 0.448).
  - The small results are in `jobs/2026-11-01_10_gen-v4/gen-v4/`. Git does not keep `examples.png`.
- Started unet-v6 on the output of gen-v4.
- unet-v6 results (unet-v4b on the real train images plus 30436 generated images, best epoch 37 of 40):
  - Val: 0.637 (unet-v4b: 0.613). C&W 0.576, MI-FGSM 0.482, PGD targeted 0.645, PGD untargeted 0.552, clean 0.92, decoy-only 0.94.
  - MI-FGSM has the largest gain: 0.482 against 0.399.
  - Generated val set: 0.518 (saliency-v1: 0.411). JPEG 80-89: 0.484, JPEG 90-95: 0.562.
  - The tune loss decreased to the end (0.805 at epoch 37). The generated images stop the overfit of unet-v3 and unet-v4b.
  - On the test set, unet-v6 answers 0,0,0,0 for 23.8% of the images (unet-v4b: 27.0%, expected: approximately 20%).
  - JPEG quality 80-88: 24.8% of 0,0,0,0 answers (unet-v4b: 29.9%). The gap between the two JPEG bands is now small.
- Decision: unet-v6 is a clear improvement (+0.024 on val, better 0,0,0,0 rates on the low JPEG quality). The submission `submission_unet-v6.csv` is ready.
- New options in `src/unet.py` for the unet-v7 runs:
  - `amp`: training in fp16 (gen-speed: approximately 3.6 times faster than fp32).
  - `ema`: an exponential average of the weights. The tune loss and the checkpoints use these weights.
  - `checkpoint_every`: a checkpoint after each n-th epoch. At the end, the checkpoint with the most tune points wins. `checkpoints.csv` has the tune and val points of each checkpoint.
  - `loss: lovasz`: BCE plus the Lovasz hinge loss, a surrogate of the mask IoU. `patch_weight` gives more weight to the patch channel.
  - `scheduler: restarts` (cosine restarts after `restart_epochs`) and `optimizer: sgd` (Nesterov, momentum 0.9).
  - `run_experiment()` gets a device, so one job trains two variants at the same time, one on each GPU.
- unet-v6 analysis: the tune loss stops at approximately 0.82 after epoch 25. The train loss is still 0.585, thus a larger UNet can help.
- unet-v4b had a lower minimum tune loss than unet-v6 (0.782 against 0.805), but fewer val points. Thus the selection of the epoch now uses the tune points.
- Jobs `2026-11-01_16_unet-v7ab` and `2026-11-01_17_unet-v7cd`. All four variants use width 64, 100 epochs, fp16, EMA and checkpoints:
  - unet-v7a: AdamW, one cycle, BCE plus Dice (the base).
  - unet-v7b: the Lovasz loss and the weight 2 on the patch channel.
  - unet-v7c: cosine restarts after each 25 epochs.
  - unet-v7d: SGD.
- unet-v6 on the leaderboard: 0.6212 (unet-v4b: 0.5590). This is the first place (the next team has 0.6024). 1188 of 5000 test answers are 0,0,0,0.
- The public half has 2500 images. With a standard deviation of approximately 0.4 for the points of one image, the standard error is approximately 0.008.
- The Kaggle dataset from the output of gen-v4 failed: "Directory already exists: gen-v4". Kaggle extracts `gen-v4.zip` to the folder `gen-v4/`, and the output already has this folder. The generator now writes `<codename>-set.zip`. The jobs read gen-v4 from the notebook output.
- New structure: one folder `jobs/` in place of `kaggle-jobs/` and `configs/`. Each job folder has `run.py`, `kernel-metadata.json`, one `<codename>.yaml` for each codename of the job, and the results in `<codename>/`. `load_config()` finds `jobs/*/<codename>.yaml`.
- `docs/results.md` now has the column Job (the job folder). The column Train now also has the tune points of the UNet methods.
- The results of a job are now in `jobs/<job folder>/output/`, with the same paths as `/kaggle/working/`. Only small files are downloaded. The root `.gitignore` keeps images, logs, `.npy` files and submission files of `output/` out of git.
- unet-v7 results (width 64, 100 epochs, fp16, EMA, checkpoint with the most tune points). All four variants are better than unet-v6 (val 0.637):

  | Variant | Change | Best checkpoint | Tune | Val | Generated val | JPEG 80-89 (generated) |
  |---|---|---|---|---|---|---|
  | unet-v7a | base (AdamW, one cycle, BCE plus Dice) | 64 | 0.655 | 0.654 | 0.535 | 0.503 |
  | unet-v7b | Lovasz loss, patch weight 2 | 79 | 0.643 | 0.646 | 0.535 | 0.504 |
  | unet-v7c | cosine restarts | 99 | 0.663 | 0.653 | 0.528 | 0.494 |
  | unet-v7d | SGD | 94 | 0.665 | 0.663 | 0.553 | 0.522 |

  - unet-v7d is the best on all four attacks (MI-FGSM 0.530, unet-v6 0.482). Clean images: 0.90 (unet-v6 0.92).
  - unet-v7a overfits after epoch 55: the tune loss increases from 0.783 to 0.879, but the tune points stay at approximately 0.65.
  - unet-v7d still improves at epoch 94-99. More epochs with SGD can help.
  - The Lovasz loss does not help (unet-v7b is the worst of the four).
  - On the test set, all variants answer 0,0,0,0 for approximately 24% of the images.
- Decision: unet-v7d is a clear improvement on val (+0.026) and on the generated val set (+0.035). The submission `submission_unet-v7d.csv` is ready.
- New options in `src/unet.py`:
  - `ensemble`: a list of codenames. The run loads their `unet.pt` from the notebook outputs and uses the mean mask. It does not train a UNet.
  - `train_seed`: changes only the start weights and the order of the images. `seed` also selects the tune part, and the generated set needs the tune part of seed 0.
- Job `2026-11-01_18_unet-v8`: unet-v8a (unet-v7d with 200 epochs) and unet-v8b (the same with train seed 1). unet-v8b shows the effect of the seed.
- Job `2026-11-01_19_unet-ens`: unet-ens1 (mean of unet-v7a, v7c and v7d) and unet-ens2 (mean of all four unet-v7 variants).
