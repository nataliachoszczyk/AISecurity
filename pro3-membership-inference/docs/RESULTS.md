# Results

Each row is one submission. The CV columns give the TPR@5%FPR of `src/validation.py` on known (2 folds x 5 seeds, mean). "All" is all known records together; its composition is not the composition of test. Test is the score on the public leaderboard.

| Codename | Job folder | Method | CV all | CV forum post | CV web photo | CV writer | CV research group | CV podcast | CV photographer | Test |
|---|---|---|---|---|---|---|---|---|---|---|
| attacks-v8a | `2026-11-27_22_attacks-v8a` | One logistic regression for each source type on the record features: LOSS, zlib, Min-K%, Min-K%++, the gain over Qwen3-0.6B-Base, over the domain reference models and over the OUT shadow model. No owner term. | 0.631 | 0.843 | 0.111 | 0.919 | 0.997 | 1.000 | 0.503 | 0.774 (#291, 2026-11-27, beats the baseline on private) |
