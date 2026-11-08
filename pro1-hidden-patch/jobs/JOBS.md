# Kaggle jobs

## How to run a job

1. Make a new folder `jobs/runs/YYYY-MM-DD_NN_<job name>` (NN: the next number after the last job in the project, not a number for each day). Put `run.py`, `kernel-metadata.json` and one `<codename>.yaml` for each codename of the job into it. Copy them from the last job.
2. Use one Kaggle notebook for each experiment: the kernel id is `<kaggle user>/bumas-p1-<codename>`. Then the link always shows the correct run. The Kaggle API does not give the version ID.
3. On a different Kaggle account, change the user in the kernel id. The account needs phone verification (for GPU and internet) and access to the dataset `jankombajn/bumas-p1-hidden-patch`. Do not upload copies of the data.
4. Settings in `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"` (2x T4), `"enable_internet": true`. The script clones the branch `pro1`, so push the code first.
5. First run on a new account: run the smoke test (`2026-10-31_01_smoke`).
6. Push with `kaggle kernels push -p jobs/runs/<job folder>`. Then run `bash jobs/wait.sh <kernel id> <job name>`. The job name is the end of the folder name. It writes each status change into the table below.
7. Download only small outputs: `kaggle kernels output <kernel id> -p jobs/runs/<job folder>/output --file-pattern '(json|csv|png)$'`. The folder `output/` has the same paths as `/kaggle/working/`. Move `submission_*.csv` to `submissions/`. Do not download `.npy`, `.zip` or `.pt` files.

One folder in `jobs/runs/` has one row. A new version of a job uses the same row and gets a note in the comment.

Status values: to run, waiting for another job, queued, running, complete, failed, cancelled. The status cell has only one of these values, because `wait.sh` changes it. Put results into the comment.

| Date | Job | Folder | Link | Comment | Status |
|---|---|---|---|---|---|
| 2026-10-31 | smoke | `2026-10-31_01_smoke` | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | Check the GPU, the data, the repository clone and the output folder. Version 1 failed: the kernel had no internet, so `git clone` failed. Version 2 was cancelled: it ran on the CPU, because the account had no phone verification. Version 3: 2x Tesla T4, Python 3.13, torch 2.11, approximately 900 images/s. | complete |
| 2026-11-01 | gen-v4 | `2026-11-01_10_gen-v4` | [bumas-p1-gen-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v4) | gen-v4: gen-v3 with a near target class for the targeted PGD and the real C&W strength. Version 1: out of memory on GPU 0. Version 2: train 30436 images, val 1100. | complete |
| 2026-11-07 | unet-v9 | `2026-11-07_22_unet-v9` | [bumas-p1-unet-v9](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v9) | unet-v9a (unet-v8b plus 6 ResNet-18 activation maps as input) and unet-v9b (unet-v8b with width 96 and 160 epochs), one on each GPU. Val: unet-v9a 0.684 (tune 0.692, best epoch 194), unet-v9b 0.669 (tune 0.670). The activation maps help. The larger UNet does not. | complete |
| 2026-11-07 | gen-v5 | `2026-11-07_24_gen-v5` | [bumas-p1-gen-v5](https://www.kaggle.com/code/jankombajn/bumas-p1-gen-v5) | gen-v5: gen-v4 with seed 5 and `base_aug` (a crop of the clean bases, brightness, contrast and color). A smoke test with 4 bases runs first in the same session. 9.5 hours for the train part. All jobs ended in 2 hours: train 31027 images, val 93. Saliency-v1 on its val: 0.430. | complete |
| 2026-11-07 | gen-v6 | `2026-11-07_27_gen-v6` | [bumas-p1-gen-v6](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-gen-v6) | Account nataliachoszczyk. gen-v5 with seed 6. The jankombajn jobs can use it only after a share of the notebook with jankombajn. Version 1 failed: the strength check of C&W with only 17 val images. Version 2 has 8 val copies: train 30703 images, val 448. Saliency-v1 on its val: 0.374. | complete |
| 2026-11-08 | unet-v12 | `2026-11-08_32_unet-v12` | [bumas-p1-unet-v12](https://www.kaggle.com/code/jankombajn/bumas-p1-unet-v12) | unet-v9a (activation maps) on the real images, gen-v4, gen-v5 and gen-v6, 100 epochs. One GPU. The inputs and masks are float16 (half_inputs), so that they fit on the GPU. Val 0.691, tune 0.698: the best single model. | complete |
| 2026-11-08 | post-v4 | `2026-11-08_37_post-v4` | [bumas-p1-post-v4](https://www.kaggle.com/code/jankombajn/bumas-p1-post-v4) | The final ensembles of the nine trained UNets (unet-v8 to unet-v12). Each model predicts each set one time. The final choice uses the tune points. Val (soft): six 0.697, all9 0.687, new7 0.696 (best tune 0.701), act2 (unet-v9a + unet-v12) 0.707 (tune 0.700, gen_val 0.601). | complete |
