# Kaggle jobs

Each job has a folder here with `run.py` and `kernel-metadata.json`. All jobs use the dataset `jankombajn/bumas-p1-hidden-patch`. `wait.sh` waits until a job stops.

Status values: to run, waiting for another job, queued, running, complete, failed, cancelled.

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-10-31 | smoke v1 | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | Check the GPU, the data, the repository clone and the output folder. The kernel had no internet, so `git clone` failed. | failed |
| 2026-10-31 | smoke v2 | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | The same check with the code in the script (no clone). It ran on the CPU, because the account had no phone verification. Replaced by v3. | cancelled |
| 2026-10-31 | smoke v3 | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | The check with `git clone`, internet and GPU T4 x2. Result: 2x Tesla T4, Python 3.13, torch 2.11, approximately 900 images/s. | complete |
| 2026-10-31 | saliency | [bumas-p1-saliency](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-saliency) | saliency-v1 (median 3x3) and saliency-v2 (Gaussian blur): tune on 1500 train images, score val, make the test submission. Run time 22 min. Val: v1 0.530, v2 0.525. | complete |
