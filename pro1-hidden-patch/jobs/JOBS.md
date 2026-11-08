# Kaggle jobs

## How to run a job

1. Make a new folder `jobs/runs/YYYY-MM-DD_NN_<job name>` (NN: the number of the job in the project). Put `run.py`, `kernel-metadata.json` and one `<codename>.yaml` for each codename of the job into it. Copy them from the last job.
2. Use one Kaggle notebook for each experiment: the kernel id is `<kaggle user>/bumas-p1-<codename>`. Then the link always shows the correct run. The Kaggle API does not give the version ID.
3. On a different Kaggle account, change the user in the kernel id. The account needs phone verification (for GPU and internet) and access to the dataset `jankombajn/bumas-p1-hidden-patch`. Do not upload copies of the data.
4. Settings in `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"` (2x T4), `"enable_internet": true`. The script clones the branch `pro1`, so push the code first.
5. Push with `kaggle kernels push -p jobs/runs/<job folder>`. Then run `bash jobs/wait.sh <kernel id> <job name>`. The job name is the end of the folder name. It writes each status change into the table below.
6. Download only small outputs: `kaggle kernels output <kernel id> -p jobs/runs/<job folder>/output --file-pattern '(json|csv|png)$'`. The folder `output/` has the same paths as `/kaggle/working/`. Move `submission_*.csv` to `submissions/`. Do not download `.npy`, `.zip` or `.pt` files.

One folder in `jobs/runs/` has one row. A new version of a job uses the same row and gets a note in the comment.

Status values: to run, waiting for another job, queued, running, complete, failed, cancelled. The status cell has only one of these values, because `wait.sh` changes it. Put results into the comment.

| Date | Job | Folder | Link | Comment | Status |
|---|---|---|---|---|---|
| 2026-11-01 | gen-v4 | `2026-11-01_10_gen-v4` | [bumas-p1-gen-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v4) | Attacked images with the patch at any position (PGD, targeted PGD to a near class, MI-FGSM, C&W; JPEG 80-95). Version 1: out of memory on GPU 0. Version 2: train 30436 images, val 1100 (gen_val of all UNet jobs). | complete |
| 2026-11-07 | unet-v9 | `2026-11-07_22_unet-v9` | [bumas-p1-unet-v9](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v9) | unet-v9a: the UNet with the integrated gradients map and 6 ResNet-18 activation maps as input, on the real images and gen-v4, 200 epochs. Val 0.684, tune 0.692, best epoch 194. | complete |
| 2026-11-07 | gen-v5 | `2026-11-07_24_gen-v5` | [bumas-p1-gen-v5](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v5) | gen-v4 with seed 5 and `base_aug` (a crop of the clean bases, brightness, contrast and color). Train 31027 images, val 93. | complete |
| 2026-11-07 | gen-v6 | `2026-11-07_27_gen-v6` | [bumas-p1-gen-v6](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-gen-v6) | gen-v5 with seed 6 on the account nataliachoszczyk. The jankombajn jobs can use it only after a share of the notebook. Version 1 failed: the strength check of C&W had only 17 val images. Version 2 has 8 val copies: train 30703 images, val 448. | complete |
| 2026-11-08 | unet-v12 | `2026-11-08_32_unet-v12` | [bumas-p1-unet-v12](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v12) | unet-v9a on the real images, gen-v4, gen-v5 and gen-v6, 100 epochs. One GPU. The inputs and masks are float16 (half_inputs), so that they fit on the GPU. Val 0.691, tune 0.698. | complete |
| 2026-11-08 | post-v4 | `2026-11-08_37_post-v4` | [bumas-p1-post-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-post-v4) | act2: the mean mask of unet-v9a and unet-v12, then the soft refinement. Tune 0.700, val 0.707, gen_val 0.601. The file `post-v4/submission_act2_soft.csv` got 0.6849 on the leaderboard. | complete |
