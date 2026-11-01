# Kaggle jobs

## How to run a job

1. Make a new folder `kaggle-jobs/YYYY-MM-DD_NN_<codename>` (NN gives the order on one day) with `run.py` and `kernel-metadata.json`. Copy them from the last job.
2. Use one Kaggle notebook for each experiment: the kernel id is `<kaggle user>/bumas-p1-<codename>`. Then the link always shows the correct run. The Kaggle API does not give the version ID.
3. On a different Kaggle account, change the user in the kernel id. The account needs phone verification (for GPU and internet) and access to the dataset `jankombajn/bumas-p1-hidden-patch`. Do not upload copies of the data.
4. Settings in `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"` (2x T4), `"enable_internet": true`. The script clones the branch `pro1`, so push the code first.
5. First run on a new account: run the smoke test (`2026-10-31_01_smoke`).
6. Push with `kaggle kernels push -p <folder>`. Then run `bash kaggle-jobs/wait.sh <kernel id> "<job name>"`. It writes each status change into the table below.
7. Download only small outputs (`kaggle kernels output ... --file-pattern`). Put `metrics.json`, the reports and the train loss into `<job folder>/<codename>/`. Do not commit submission files: the repository is public.

Status values: to run, waiting for another job, queued, running, complete, failed, cancelled.

| Date | Job | Folder | Link | Comment | Status |
|---|---|---|---|---|---|
| 2026-10-31 | smoke v1 | `2026-10-31_01_smoke` | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | Check the GPU, the data, the repository clone and the output folder. The kernel had no internet, so `git clone` failed. | failed |
| 2026-10-31 | smoke v2 | `2026-10-31_01_smoke` | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | The same check with the code in the script (no clone). It ran on the CPU, because the account had no phone verification. Replaced by v3. | cancelled |
| 2026-10-31 | smoke v3 | `2026-10-31_01_smoke` | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | The check with `git clone`, internet and GPU T4 x2. Result: 2x Tesla T4, Python 3.13, torch 2.11, approximately 900 images/s. | complete |
| 2026-10-31 | saliency | `2026-10-31_02_saliency` | [bumas-p1-saliency](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-saliency) | saliency-v1 (median 3x3) and saliency-v2 (Gaussian blur): tune on 1500 train images, score val, make the test submission. Run time 22 min. Val: v1 0.530, v2 0.525. | complete |
| 2026-10-31 | unet v1 | `2026-10-31_03_unet-v1` | [bumas-p1-unet, version 1](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-unet) | unet-v1: train the UNet (60 epochs), tune on 450 train images, score val, make the test submission. Val 0.350, mask IoU 0.21: the UNet does not find the weak noise well. | complete |
| 2026-10-31 | unet v2 | `2026-10-31_04_unet-v2` | [bumas-p1-unet, version 2](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-unet?scriptVersionId=364350786) | unet-v2: the UNet also gets the integrated gradients map of ResNet-18. Val 0.508, mask IoU 0.33. | complete |
| 2026-10-31 | unet v3 | `2026-10-31_05_unet-v3` | [bumas-p1-unet-v3](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v3) | unet-v3: unet-v2 with 200 epochs and the loss on the tune part. First job on the account `jankombajn`. | complete |
| 2026-10-31 | gen v1 | `2026-10-31_06_gen-v1` | [bumas-p1-gen-v1](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v1) | gen-v1: generate attacked images with patches at any position (train bases x16, val bases x8). Score saliency-v1 on the generated val set. | cancelled |
| 2026-10-31 | gen v2 | `2026-10-31_07_gen-v2` | [bumas-p1-gen-v2](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v2) | Data leakage: it used the bases of the UNet tune part. Replaced by gen-v3. gen-v2: the one large generated set (no rotations, class check, stop after 9 hours, one zip for the dataset). | cancelled |
| 2026-10-31 | unet v4 | `2026-10-31_08_unet-v4` | [bumas-p1-unet-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v4) | unet-v4a (old boxes) and unet-v4b (fitted boxes of an allowed size), both 80 epochs with the weights of the best epoch. Val: v4a 0.448, v4b 0.613. | complete |
| 2026-10-31 | gen v3 | `2026-10-31_09_gen-v3` | [bumas-p1-gen-v3](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v3) | Replaced by gen-v4 (targeted PGD failed with a random target). gen-v3: gen-v2 without the bases of the UNet tune part. The one set for the Kaggle dataset. | cancelled |
| 2026-11-01 | gen v4 | `2026-11-01_10_gen-v4` | [bumas-p1-gen-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v4) | gen-v4: gen-v3 with a near target class for the targeted PGD and the real C&W strength. The one set for the Kaggle dataset. | failed (version 1: out of memory on GPU 0), version 2 to run |
| 2026-11-01 | gen smoke | `2026-11-01_11_gen-smoke` | [bumas-p1-gen-smoke](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-smoke) | Smoke test of gen-v4 on the CPU: the full flow with 4 bases for each part. Passed. C&W strength 17-20/255 (too large). | complete |
| 2026-11-01 | gen speed | `2026-11-01_12_gen-speed` | [bumas-p1-gen-speed](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-speed) | Speed of one PGD attack: fp32 or fp16, 1 GPU or 2 GPUs, batch 256, 512 or 1024. Version 1: out of memory at batch 1024 in fp32. | to run |
| 2026-11-01 | unet v5 | `2026-11-01_13_unet-v5` | [bumas-p1-unet-v5](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v5) | unet-v5: unet-v4b plus 24 candidates from the integrated gradients map. Val 0.592 (worse than unet-v4b). | complete |
| 2026-11-01 | jpeg quality | `2026-11-01_14_jpeg-quality` | [bumas-p1-jpeg-quality](https://www.kaggle.com/code/jankombajn/bumas-p1-jpeg-quality) | CPU: estimate the JPEG quality of all train and test images. Test: 20% have quality 80-88, train has only 90-96. | complete |
| 2026-11-01 | unet v6 | `2026-11-01_15_unet-v6` | [bumas-p1-unet-v6](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v6) | unet-v6: unet-v4b on the real and the generated train images (dataset bumas-p1-gen). | waiting for another job |
