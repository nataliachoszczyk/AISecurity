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
- Added `kaggle-jobs/` with `JOBS.md` (job list and status) and the job scripts. The smoke test found that Kaggle runs Python 3.13.
- Ran the job `saliency` on Kaggle (2x T4, 22 min). Val: saliency-v1 (median 3x3) 0.530, saliency-v2 (Gaussian blur) 0.525. Clean 0.88, decoy-only 0.94, patch images 0.30-0.51 (MI-FGSM is the worst).
- The weak part is the box: the best of the 32 candidates has a mean IoU of only 0.58 on patch images. Maps with only the positive part of the gradients were worse (0.35).
- Made `submissions/submission_saliency-v1.csv` for the leaderboard: 20.9% of the answers are 0,0,0,0, and 51% of the boxes are in the top half.
- Added `src/unet.py`: a UNet on the image and its noise residual predicts the masks of the patch and the decoys. The causal check with ResNet-18 selects the patch. Flips and 90 degree rotations are the augmentation, and the prediction is the mean over all 8.
- unet-v1 (image and residual only): val 0.350, mask IoU 0.21. The UNet does not find the weak noise well from 4000 images.
- unet-v2 (with the integrated gradients map as input): val 0.508, mask IoU 0.33. Detection is almost perfect (clean 1.00, decoy-only 0.98), but the boxes are worse than in saliency-v1 (0.530).
- Changed `kaggle-jobs/wait.sh`: it writes each status change of a job into `JOBS.md`.
- Added the small outputs of each Kaggle run to `kaggle-jobs/<job folder>/<codename>/`: `metrics.json`, `report_*.csv` and the train loss. Submission files stay out of git, because the repository is public.
- `src/unet.py` now writes the loss of each epoch to `train_log.csv`.

### Conclusions on 2026-10-31

- The best method is saliency-v1 (val 0.530). The file `submission_saliency-v1.csv` is ready, but nobody sent it to the leaderboard yet. Git does not keep it. Download it from the output of the Kaggle notebook `bumas-p1-saliency`, or run the job again.
- Detection (patch or no patch) is good in all methods. The unet-v2 detection is almost perfect (clean 1.00, decoy-only 0.98).
- The box is the weak part. MI-FGSM patches get the lowest points in all methods (0.21-0.32).
- The UNet loss still decreased at epoch 60 (`kaggle-jobs/2026-10-31_0[34]_unet-v[12]/unet-v*/train_log.txt`). Thus the training was too short.
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
