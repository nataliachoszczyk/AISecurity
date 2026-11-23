# AI/ML Security

AI models now make decisions in many systems: cameras, spam filters, medical tools and cars. These models have weak points. A small change to the input, too small for a human to see, can make a model give a wrong answer. An attacker can also poison the data that a model learns from, or copy a model.

This repository has our projects about these problems. In each project, we examine one security problem of AI models in practice.

## Table of Contents

- [Project 1 - Hidden Adversarial Patch](#project-1---hidden-adversarial-patch)
- [Project 2 - Backdoor Detection](#project-2---backdoor-detection)
- Project 3 - TBD
- Project 4 - TBD
- Project 5 - TBD
- Project 6 - TBD

---

## Project 1 - Hidden Adversarial Patch

> Full report: [`pro1-hidden-patch/README.md`](pro1-hidden-patch/README.md)

An image of 64 × 64 pixels can hide a rectangle of weak adversarial noise. This patch makes ResNet-18 give a wrong class. Decoys have random noise of the same strength, but they do not change the class. The task is to find the box of the patch, or to answer `0,0,0,0` if the image has no patch.

![Examples](pro1-hidden-patch/assets/example.png)

*Orange: the true patch. Blue: decoys. Green: our answer. The noise is in all rectangles, but our UNet marks only the patch.*

### Solution

1. **More data.** All train patches are in the bottom half, but approximately half of the test patches are in the top half. We attacked ResNet-18 ourselves and made 92,166 new images with patches at all positions.
2. **Hints from ResNet-18.** Gradient and activation maps show where the noise pushes the answer of ResNet-18.
3. **Two UNets.** From the image and the hints, two UNets draw a patch map and a decoy map.
4. **Final check.** For the best candidate boxes, we remove the noise only inside the box and ask ResNet-18 again. Only the true patch changes the class. If no box changes it, the answer is `0,0,0,0`.

### Results

The score is the mean IoU (a correct `0,0,0,0` gets 1 point).

| Method                         | Val   | Gen val |
|--------------------------------|:-----:|:-------:|
| Always `0,0,0,0`               | 0.200 | 0.000   |
| unet-v9a only                  | 0.686 | 0.588   |
| unet-v12 only                  | 0.695 | 0.598   |
| **Final: unet-v9a + unet-v12** | **0.707** | **0.601** |

---

## Project 2 - Backdoor Detection

> Full report: [`pro2-backdoor-detection/README.md`](pro2-backdoor-detection/README.md)

We get 200 ResNet-18 classifiers for 20 classes from a supplier that is not trusted. Some models have 1-3 backdoors: an image with a trigger goes to the class that the attacker selected. For each model and class, we give a number that tells how much we suspect this class. Only 6 train models have labels.

![Hit tables](pro2-backdoor-detection/assets/hit_matrix.png)

*Left: with one of our probes, images of all classes go to the backdoor target (red column). Middle: a clean model with the same probe. Right: a targeted backdoor moves only one source class.*

### Solution

1. **Probe bank.** We apply the same 196 changes (patches, blends, stripes, warps, colors, noise and more) to 400 clean images. A backdoor is a wide shortcut, so a similar change also activates it.
2. **Scan.** For each model, probe and pair of classes, we measure which part of the images moves to the class.
3. **Siblings.** All models have the same base and data. Thus we compare each value with the same value in all other models (robust z-score). A backdoor class is far from the same class in the other models.
4. **Score.** The mean of four parts: two measures (images that move, change of the log probability) and two aggregations (all source classes, one source class).
5. **Own validation models.** We made 183 valid models with known backdoors in two sets. Set B has only trigger types that we did not use to design the probe bank.

### Results

The score is the ROC-AUC of all (model, class) pairs.

| Method | Set A | Set B (unknown triggers) | Leaderboard (public) |
|---|:---:|:---:|:---:|
| Neural Cleanse (baseline) | - | - | 0.901 |
| **Final: probe bank, `combo4_z`** | **0.991** | **0.9998** | **0.996** |
