# Results

This table shows the methods for the leaderboard. A method gets its row before the submission; the test score follows later. The results of all runs are in `jobs/runs/<job folder>/output/<codename>/` and in `jobs/JOBS.md`. The split is `splits/split.csv` (4500 train, 500 val, seed 0). Val mean is the mean of Val and Val flip. Job is the folder in `jobs/runs/` with the config (`<codename>.yaml`) and the results.

Train gives the points on the train images that selected the thresholds. For saliency-v1, these are all 4500 train images. For a UNet, these are the 450 images of the tune part. The UNet does not train on them.

Val flip is not valid at this time. A vertical flip makes ResNet-18 misclassify the image, and the flipped patch is not the cause. Use Val to compare methods until a val set with generated attacks in the top half is available.

| Date | Codename | Job | Description | Train | Val | Val flip | Val mean | Test (leaderboard) |
|---|---|---|---|---|---|---|---|---|
| 2026-10-31 | saliency-v1 | `2026-10-31_02_saliency` | Integrated gradients to a 3x3 median image, then a causal check of 32 candidate boxes with ResNet-18. | 0.548 | 0.530 | 0.246 | 0.388 | 0.5373 |
| 2026-11-01 | unet-v4b | `2026-10-31_08_unet-v4` | UNet (image, residual, integrated gradients map) predicts the patch and decoy masks. Candidates are the best rectangles of an allowed size on the masks. The causal check with ResNet-18 selects the patch. | 0.608 | 0.613 | 0.275 | 0.444 | 0.5590 |
| 2026-11-01 | unet-v4b-saliency | - (a local join of two submission files) | unet-v4b, but the saliency-v1 box where only unet-v4b answers 0,0,0,0 (7% of the test images). No val score: the val answers of each image were not saved. | - | - | - | - | 0.5648 |
| 2026-11-01 | unet-v6 | `2026-11-01_15_unet-v6` | unet-v4b, trained on the real train images plus 30436 generated images (gen-v4: patches at all positions, JPEG 80-95). Generated val: 0.518. | 0.636 | 0.637 | 0.315 | 0.476 | 0.6212 |
| 2026-11-01 | unet-v7d | `2026-11-01_17_unet-v7cd` | unet-v6 with width 64, 100 epochs, fp16, EMA weights and SGD. The checkpoint with the most tune points (epoch 94) wins. Generated val: 0.553. | 0.665 | 0.663 | 0.338 | 0.501 | pending |
