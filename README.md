AI & Machine Learning Security

  
**Framework:** PyTorch | **Language:** Python

Each project is in a separate folder. Each folder has a README with the full report.

---

## Table of Contents

- [Project 1 - Hidden Adversarial Patch](#project-1---hidden-adversarial-patch)
- Project 2 - TBD
- Project 3 - TBD
- Project 4 - TBD
- Project 5 - TBD
- Project 6 - TBD

---

## Project 1 - Hidden Adversarial Patch

> Full report: [`pro1-hidden-patch/README.md`](pro1-hidden-patch/README.md)

Each 64 × 64 image can contain one rectangle with weak adversarial noise against ResNet-18. Some images also contain decoys: rectangles with random noise that do not change the prediction. The task is to find the box of the adversarial patch, or to answer `0,0,0,0` if the image has no patch. The score is the mean IoU.

| Method              | Score ↑ |
|---------------------|:-------:|
| All `0,0,0,0`       | 0.2000  |
| Course baseline     | 0.2200  |
| **Our solution**    | TBD     |

---

## Setup

The projects use one virtual environment in the root folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local machines have no GPU. Long computations run on Kaggle. The data of each project is a private Kaggle dataset.
