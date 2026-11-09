# AI & Machine Learning Security

  
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

What we did:

- The train patches are only in the bottom half, but approximately half of the test patches are in the top half. Thus we attacked ResNet-18 ourselves and made 92,166 new images with the patch at any position.
- Two UNets draw a map of the patch. Their input is the image and maps from ResNet-18 that show where the noise changes the decision.
- For the best boxes on the map, we remove the noise and run ResNet-18 again. Only the true patch changes the class. If no box changes it, the answer is `0,0,0,0`.

Result on our validation set (500 train images that no step used for training or tuning):

| Method              | Score ↑ |
|---------------------|:-------:|
| All `0,0,0,0`       | 0.200   |
| **Our solution**    | **0.707** |
