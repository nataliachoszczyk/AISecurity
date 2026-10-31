# Kaggle jobs

Each job has a folder here with `run.py` and `kernel-metadata.json`. All jobs use the dataset `jankombajn/bumas-p1-hidden-patch`. `wait.sh` waits until a job stops.

Status values: to run, waiting for another job, queued, running, complete, failed.

| Date | Job | Link | Comment | Status |
|---|---|---|---|---|
| 2026-10-31 | smoke v1 | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | Check the GPU, the data, the repository clone and the output folder. The kernel had no internet, so `git clone` failed. | failed |
| 2026-10-31 | smoke v2 | [bumas-p1-smoke](https://www.kaggle.com/code/nataliachoszczyk/bumas-p1-smoke) | The same check with the code from `bundle.py` (no clone). | running |
