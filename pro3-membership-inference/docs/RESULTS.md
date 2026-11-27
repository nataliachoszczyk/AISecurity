# Results

Each row is one submission. The CV columns give the TPR@5%FPR of `src/validation.py` on known (2 folds x 5 seeds, mean). "All" is all known records together; its composition is not the composition of test. Test is the score on the public leaderboard.

| Codename | Job folder | Method | CV all | CV forum post | CV web photo | CV writer | CV research group | CV podcast | CV photographer | Test |
|---|---|---|---|---|---|---|---|---|---|---|
| attacks-v8a | `2026-11-27_22_attacks-v8a` | One logistic regression for each source type on the record features: LOSS, zlib, Min-K%, Min-K%++, the gain over Qwen3-0.6B-Base, over the domain reference models and over the OUT shadow model. No owner term. | 0.631 | 0.843 | 0.111 | 0.919 | 0.997 | 1.000 | 0.503 | 0.774 (#291, 2026-11-27, beats the baseline on private) |
| attacks-v11 | `2026-11-27_26_attacks-v10b` + `2026-11-27_66_forum-final` | The score of attacks-v10b with the forum post log-odds of the job forum-final. Set types: the owner mixture of `src/photographer.py` (each owner is a victim or clean, EM on the test records). Forum posts: v6 features, forum shadow models and length features. Web photos: v6 features with C = 0.01. Both shadow pairs. Global simulation on known: 0.813. Test-like simulation: 0.867 (std 0.018, attacks-v10c, without the forum module). | 0.643 | 0.871 | 0.147 | 0.922 | 0.994 | 1.000 | 0.639 | pending |
