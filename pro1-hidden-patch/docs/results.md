# Results

This table shows only the methods that went to the leaderboard. The results of all runs are in `kaggle-jobs/<job folder>/<codename>/` and in `kaggle-jobs/JOBS.md`. The split is `splits/split.csv` (4500 train, 500 val, seed 0). Val mean is the mean of Val and Val flip.

Val flip is not valid at this time. A vertical flip makes ResNet-18 misclassify the image, and the flipped patch is not the cause. Use Val to compare methods until a val set with generated attacks in the top half is available.

| Date | Codename | Description | Train | Val | Val flip | Val mean | Test (leaderboard) |
|---|---|---|---|---|---|---|---|
| 2026-10-31 | saliency-v1 | Integrated gradients to a 3x3 median image, then a causal check of 32 candidate boxes with ResNet-18. | 0.548 | 0.530 | 0.246 | 0.388 | 0.5373 |
| 2026-11-01 | unet-v4b | UNet (image, residual, integrated gradients map) predicts the patch and decoy masks. Candidates are the best rectangles of an allowed size on the masks. The causal check with ResNet-18 selects the patch. | - | 0.613 | 0.275 | 0.444 | 0.5590 |
