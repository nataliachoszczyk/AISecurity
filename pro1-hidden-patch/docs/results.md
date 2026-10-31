# Results

This table shows only the methods that went to the leaderboard. The split is `splits/split.csv` (4500 train, 500 val, seed 0). Val mean is the mean of Val and Val flip.

Val flip is not valid at this time. A vertical flip makes ResNet-18 misclassify the image, and the flipped patch is not the cause. Use Val to compare methods until a val set with generated attacks in the top half is available.

| Date | Codename | Description | Train | Val | Val flip | Val mean | Test (leaderboard) |
|---|---|---|---|---|---|---|---|
