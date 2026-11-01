# Experiments on Kaggle

This document describes a method to run experiments with models on Kaggle from a local repository. It is a collection of rules and lessons from real projects. A project can use only the parts that it needs, and change them for its own work.

The local machine has the code and the records. Kaggle does the heavy computation. Git connects the two.

## 1. Local setup

The local machine needs these items:

| Item | Where | What to do |
|---|---|---|
| Kaggle account | kaggle.com | Do the phone verification (Settings). Without it, a kernel gets no GPU and no internet. |
| Python environment | `.venv` in the root folder of the repository | Make it with uv (see the README or the instructions of the repository). |
| Kaggle CLI | the package `kaggle` in `requirements.txt` | Install it into `.venv`. Version 2.2.5 works. Then the command is `.venv/bin/kaggle`. |
| API token | `~/.kaggle/access_token` | On kaggle.com, open Settings, API, and generate a new token. Write the token into this file as one line. |
| Access to the data | the private Kaggle dataset of the project | The owner of the dataset shares it with the account. |
| Git remote | a public repository with the project branch | The kernels clone the code from it (section 4). |

Do these steps:

1. Install the environment:

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv -r requirements.txt
   ```

2. Write the token into the file and protect the file:

   ```bash
   mkdir -p ~/.kaggle
   echo "<token>" > ~/.kaggle/access_token
   chmod 600 ~/.kaggle/access_token
   ```

3. Do a test of the connection. The command shows the kernels of the account:

   ```bash
   .venv/bin/kaggle kernels list --mine
   ```

Do not commit the token. It gives full access to the account.

## 2. Platform facts

| Item | Value |
|---|---|
| GPU machine | `"machine_shape": "NvidiaTeslaT4"` gives 2x Tesla T4, 15 GB of memory each. |
| GPU sessions at the same time | 2 for each account. A third push fails with "Maximum batch GPU session count of 2 reached". |
| GPU time | A weekly quota (approximately 30 hours). A session counts its hours, not the number of its GPUs. |
| CPU sessions | 4 cores and approximately 30 GB of RAM. They do not use the GPU quota. |
| Session limit | 12 hours. |
| Output folder | `/kaggle/working/`, approximately 20 GB. Only this folder is kept after the session. |
| Temporary folder | `/kaggle/temp/`. It is deleted after the session. |
| Input data | `/kaggle/input/<slug>/...`, read-only. |
| Python | 3.13 (2026). The local environment can differ. Do not use features of one version only. |
| Account | GPU and internet need a phone verification of the account. Without it, a GPU kernel runs on the CPU and no error occurs. |

Important behavior:

- If a kernel fails (status ERROR), Kaggle deletes its complete output. Hours of results can be lost.
- The CLI cannot stop a kernel. Only the web page can cancel it.
- The CLI does not give the version ID of a run. Thus use one kernel (notebook) for each job, not many versions of one kernel.
- `kaggle kernels logs -f <user>/<project>-<job name>` shows the live log. The output of child processes is sometimes not in it.
- Use `kaggle kernels status <user>/<project>-<job name>` for the state: QUEUED, RUNNING, COMPLETE, ERROR or CANCEL.

## 3. Repository structure

```
<project>/
  src/                    all logic (Python modules)
  jobs/
    JOBS.md               the list of all jobs and their status
    wait.sh               follows one kernel and writes its status into JOBS.md
    runs/
      YYYY-MM-DD_NN_<job>/  one folder for each job (NN: the number of the job in the project)
        run.py              the script that Kaggle runs
        kernel-metadata.json
        <codename>.yaml     one config for each experiment of the job
        output/             the small results, with the same paths as /kaggle/working/
          <codename>/       metrics.json, report_*.csv, train_log.csv, ...
```

### Names

The job folder `YYYY-MM-DD_NN_<job name>` identifies a job. NN is the number of the job in the project (01, 02, 03, ...). It does not start again on a new day. Thus the folders sort in the order of the jobs, and the number alone identifies a job. One job folder is one Kaggle kernel and one row in `JOBS.md`.

| Name | Where it occurs |
|---|---|
| Job folder, for example `2026-11-01_16_unet-v7ab` | The folder in `jobs/runs/` and the column Folder in `JOBS.md`. |
| Job name: the last part of the job folder, for example `unet-v7ab` | The kernel id (`<user>/<project>-unet-v7ab`), the column Job in `JOBS.md`, the second argument of `wait.sh`. |
| Codename, for example `unet-v7a` | The config file (`unet-v7a.yaml`), the output folder (`output/unet-v7a/`), and a table of results, if the project has one. |

- Each name is unique in the project. A search for a name finds all files of this job or experiment.
- A job has one or more codenames: one config file and one output folder for each codename.
- Variants of one idea get letters: `unet-v7a`, `unet-v7b`, `unet-v7c`, `unet-v7d`.
- The job name is the common part of its codenames plus their letters. The job `unet-v7ab` runs `unet-v7a` and `unet-v7b`, and the job `unet-v7cd` runs `unet-v7c` and `unet-v7d`.
- If a job has only one experiment, its codename is the job name.
- Use only lowercase letters, digits and hyphens. Kaggle makes the kernel slug from the title, and other characters change.

### Rules

- A job is one Kaggle kernel. A job can run more than one experiment (codename), for example one on each GPU.
- The config of an experiment is in the folder of the job that runs it. There is no separate folder for configs.
- The folder of a job does not change after the start. A new variant gets a new folder.
- The job folders are in `jobs/runs/`. Thus `jobs/` shows only `JOBS.md`, `wait.sh` and `runs/`, also when there are many jobs.
- `output/` keeps only small files. Data, model weights and caches stay on Kaggle.

A function finds a config from its codename:

```python
RUNS_DIR = PROJECT_DIR / "jobs" / "runs"

def load_config(codename: str) -> dict:
    matches = sorted(RUNS_DIR.glob(f"*/{codename}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one config for {codename}, found {matches}")
    cfg = yaml.safe_load(matches[0].read_text())
    cfg["codename"] = codename
    return cfg
```

Lines for `.gitignore`. They keep data, model weights and logs out of git:

```
**/jobs/runs/*/output/**/*.npy
**/jobs/runs/*/output/**/*.npz
**/jobs/runs/*/output/**/*.log
*.pt
*.zip
```

## 4. Code and data on Kaggle

### Code

The kernel clones the project branch from a public repository. Thus every run uses code from git, and the commit hash goes into the log. Push the code before the push of the kernel.

If the repository has more than one project, clone only the project folder with a sparse checkout. The templates below do this. If the project is the complete repository, clone the complete repository and remove the `sparse-checkout` line:

```python
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, CLONE], check=True)
sys.path.insert(0, CLONE)
```

Clone into `/tmp`, not into `/kaggle/working`. Otherwise the clone becomes a part of the output.

### Data

- Keep the data in a private Kaggle dataset. Upload the original zip as one file, not thousands of small files. Share the dataset with the team.
- The output of one kernel can be the input of another kernel: `"kernel_sources": ["<user>/<project>-<job name>"]`. Thus a data set or model weights from one job go to the next job without a download.
- The path under `/kaggle/input` can change. Find files with a recursive search, for example `Path("/kaggle/input").rglob("val.npy")`.
- To make a dataset from a kernel output, use the page of the kernel: Output, New Dataset. Kaggle extracts a zip to a folder with the name of the zip. If the output also has a folder with this name, the creation fails ("Directory already exists"). Thus give a zip a name like `<codename>-set.zip`.

### `kernel-metadata.json`

```json
{
  "id": "<user>/<project>-<job name>",
  "title": "<project>-<job name>",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["<user>/<dataset>"],
  "competition_sources": [],
  "kernel_sources": []
}
```

For a CPU job, use `"enable_gpu": false` and remove `machine_shape`.

### `run.py` for one experiment on one GPU

```python
"""<What the job does, in one or two sentences.>"""
import subprocess
import sys

REPO = "https://github.com/<owner>/<repository>.git"
BRANCH = "<project branch>"
PROJECT = "<project folder>"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAMES = ["<codename>"]

import torch

# A long job on the CPU wastes hours, so stop immediately without a GPU.
assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src.config import load_config
from src.experiment import run_experiment

for codename in CODENAMES:
    run_experiment(load_config(codename))
```

## 5. Use of the two GPUs

A T4 x2 session has two GPUs, and the quota counts the session. A job that uses one GPU wastes half of its cost.

One process on each GPU uses both GPUs. Each process does its own work: one experiment, or one half of the data (shard).

### `run.py` for two experiments, one on each GPU

```python
"""Run two experiments at the same time, one on each GPU."""
import subprocess
import sys

REPO = "https://github.com/<owner>/<repository>.git"
BRANCH = "<project branch>"
PROJECT = "<project folder>"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAMES = ["<codename a>", "<codename b>"]


def run(codename: str, gpu: int) -> None:
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.config import load_config
    from src.experiment import run_experiment
    run_experiment(load_config(codename), f"cuda:{gpu}")


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() == 2, "this job needs 2 GPUs (T4 x2)"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    with torch.multiprocessing.get_context("spawn").Pool(2) as pool:
        pool.starmap(run, [(codename, gpu) for gpu, codename in enumerate(CODENAMES)])
```

Rules for this pattern:

- Use the `spawn` context. CUDA does not work in a process from `fork`.
- Put the main code under `if __name__ == "__main__":`. Each new process imports `run.py` again.
- The experiment function gets the device as an argument, and it calls `torch.cuda.set_device(device)`.
- Each process writes its results into its own folder, `/kaggle/working/<codename>/`.
- The main process must not keep GPU memory. If it used the GPU before the start of the pool, call `torch.cuda.empty_cache()`. In one job, the main process kept 12 GB of cache on GPU 0, and the process for GPU 0 failed with "CUDA out of memory".
- If one process fails, the complete job fails, and Kaggle deletes the output of both experiments. Thus do a smoke test first (section 7).
- Two jobs with two experiments each give four experiments at the same time.
- Two processes share the RAM of the session. Each process must stay below approximately half of it.

For a CPU job, use `multiprocessing.Pool(4)` for the 4 cores.

## 6. Start and follow a job

### Start

1. Commit and push the code.
2. Add a row to `jobs/JOBS.md` with the status `to run`.
3. Push the kernel and start `wait.sh` for it.

The push sends `run.py` and `kernel-metadata.json` of the job folder to Kaggle and starts the kernel:

```bash
# From the project folder:
kaggle kernels push -p jobs/runs/YYYY-MM-DD_NN_<job name>
```

Output if the push is correct:

```
Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/<user>/<project>-<job name>
```

The first push makes the kernel. A second push of the same folder makes version 2 of the same kernel.

If two GPU sessions are active, the push fails. This loop tries again until a slot is free, and then follows the job:

```bash
until out=$(kaggle kernels push -p jobs/runs/YYYY-MM-DD_NN_<job name> 2>&1); echo "$(date +%H:%M) $out" | tail -1; echo "$out" | grep -q "successfully pushed"; do sleep 90; done; bash jobs/wait.sh <user>/<project>-<job name> <job name>
```

### `jobs/wait.sh`

```bash
#!/usr/bin/env bash
# Follow a Kaggle kernel until it stops. Write each status change into JOBS.md.
# Usage: bash wait.sh <owner/kernel-slug> "<job name in the JOBS.md row>"
DIR="$(dirname "$0")"
KAGGLE="$DIR/../../.venv/Scripts/kaggle"
[ -x "$KAGGLE" ] || [ -x "$KAGGLE.exe" ] || KAGGLE="$DIR/../../.venv/bin/kaggle"  # Linux and macOS
last=""
while true; do
  raw=$("$KAGGLE" kernels status "$1" 2>&1)
  case "$raw" in
    *QUEUED*) status="queued" ;;
    *RUNNING*) status="running" ;;
    *COMPLETE*) status="complete" ;;
    *ERROR*) status="failed" ;;
    *CANCEL*) status="cancelled" ;;
    *) status="$last" ;;
  esac
  if [ -n "$status" ] && [ "$status" != "$last" ]; then
    sed -i "/^| [0-9-]* | $2 |/ s/| [a-z ]* |\$/| $status |/" "$DIR/JOBS.md"
    echo "$(date +%H:%M) $status"
    last="$status"
  fi
  case "$status" in complete|failed|cancelled) echo "$raw"; exit 0 ;; esac
  sleep 60
done
```

The path to the `kaggle` command assumes a `.venv` two levels above `jobs/`. Change it for a different structure.

The job folder `jobs/runs/YYYY-MM-DD_NN_<job name>/` has the kernel `<user>/<project>-<job name>` and the row `<job name>` in `JOBS.md`. The call:

```bash
# From the project folder, after the push of the kernel:
bash jobs/wait.sh <user>/<project>-<job name> <job name>
```

Output: one line for each status change. The last line is the final status from Kaggle.

```
13:48 running
18:52 complete
<user>/<project>-<job name> has status "KernelWorkerStatus.COMPLETE"
```

Run it as a background task, because it waits until the kernel stops (minutes or hours).

### Rules for the follow-up

- Start `wait.sh` after each push: for a new job and also for a new version of an old job. Without it, the status in `JOBS.md` stays old.
- Use one background task for each job. A tool that waits for a background task gives a notice only when the complete task stops. If one task follows two jobs, the end of the first job stays without a notice until the second job ends.
- If a `wait.sh` process stops (for example after a move of the folder), start it again.
- When a job ends, check that its row shows the final status.

## 7. Protect long runs

- **Smoke test.** Before a run of many hours, run the same code path on a small part of the data. Use a CPU kernel, or the local machine if it has enough memory. The test must do all steps, also the final steps (selection, reports, final files). An error at the end of a long run loses all its results.
- **Early checks.** Put cheap checks at the start. If a check fails, stop the job. Example: after a small part of a data generation, compare a statistic of the new data with the expected value. If it is wrong, the job stops before the long part.
- **Deadline.** A long loop gets a deadline (for example 9 hours). After it, no new batch starts, and the job writes its results before the 12-hour limit.
- **No heavy work on the local machine.** The local machine does only small checks. It can stop a process if the memory is low.

## 8. Outputs

1. List the files and their sizes: `kaggle kernels files <user>/<project>-<job name>`.
2. Download only the small files into the output folder of the job:

   ```bash
   kaggle kernels output <user>/<project>-<job name> -p jobs/runs/YYYY-MM-DD_NN_<job name>/output --file-pattern '(json|csv|png)$'
   ```

   The pattern is a regular expression for the complete path. Do not start it with `^`.
3. Delete the downloaded `.log` file if it is not necessary.
4. Commit the small results.

Write large intermediate data only to `/kaggle/temp/` or to `/kaggle/working/cache/`. Write small results as CSV or JSON.

## 9. Records

### `jobs/JOBS.md`

The file starts with a short procedure (sections 6 and 8), then the table:

```markdown
Status values: to run, waiting for another job, queued, running, complete, failed, cancelled. The status cell has only one of these values, because `wait.sh` changes it. Put results into the comment.

| Date | Job | Folder | Link | Comment | Status |
|---|---|---|---|---|---|
| YYYY-MM-DD | <job name> | `YYYY-MM-DD_NN_<job name>` | [<project>-<job name>](https://www.kaggle.com/code/<user>/<project>-<job name>) | <what the job does; the main results after the end> | complete |
```

- The column Job is the name that `wait.sh` gets as its second argument.
- The status cell has one word only. A result in this cell stops `wait.sh`, because its pattern does not match.
- One job folder has one row. A new version of the job (for example after an error) uses the same row and gets a note in the comment, for example "Version 1: out of memory on GPU 0".
- A second row of the same job causes errors: `wait.sh` changes all rows with the job name.

### A table of results (optional)

Some projects need a table that compares the methods. The columns depend on the project, so decide them at the start of the project. If the project has such a table, these rules help:

- Each row has the codename and the job folder. Then the config and the results of a row are easy to find.
- Write a row as soon as its results are available. Write "pending" only for a value that does not exist yet.
- Text above the table tells what each column measures.

## 10. Experiment method

- **Codenames.** Each experiment has a codename with a version, for example `unet-v7a`. Variants of one idea get letters (section 3).
- **One change for each variant.** A group of variants has a common base, and each variant changes one thing. Then the difference shows the effect of this change.
- **Split.** Keep a fixed validation split in a file. Keep a small tune part of the train data for thresholds and for the selection of the checkpoint. Use the validation set only for decisions.
- **Checkpoints.** Train longer and keep checkpoints (for example after each 5 epochs, in CPU memory). At the end, score each checkpoint with the real metric on the tune part, and keep the best. The loss does not always follow the metric.
- **Noise.** Two runs with the same settings and a different seed show the random difference. A difference smaller than this is not a result. Keep a separate `train_seed` if the main seed also selects the data split.
