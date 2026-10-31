# Project 1 - Hidden Adversarial Patch

## Task

The images have a size of 64 × 64 pixels. On some images, the attacker added adversarial noise to one rectangle (the patch). The noise makes the ResNet-18 model in `model/` predict a wrong class. The noise is weak and usually not visible.

Some images also contain decoys. A decoy is a rectangle with random noise of the same strength. A decoy does not change the prediction of the model.

The split of images is the same in the train set and in the test set:

| Image type                  | Share |
|-----------------------------|:-----:|
| Patch (sometimes with decoys) | 80%  |
| Decoys only                 | 10%   |
| Clean                       | 10%   |

Some test images are more difficult than the train images.

For each test image, the submission gives the box `x_min,y_min,x_max,y_max` of the patch, or `0,0,0,0` if the image has no patch. The box uses the same order as `img[y_min:y_max, x_min:x_max]`.

![Box](assets/ramka.svg)

## Scoring

| True image   | Answer     | Points |
|--------------|------------|:------:|
| No patch     | `0,0,0,0`  | 1      |
| No patch     | box        | 0      |
| Patch        | `0,0,0,0`  | 0      |
| Patch        | box        | IoU    |

A box on a decoy gets 0 points. The final score is the mean of the points.

![IoU](assets/iou.svg)

## Data

| Item                      | Content |
|---------------------------|---------|
| `model/`                  | ResNet-18 (`resnet.py`, `resnet18_tin.pt`, 200 Tiny ImageNet classes). The input is `(N, 3, 64, 64)` in `[0, 1]`. |
| `train/images/`, `train_labels.csv` | 5000 images with true boxes, attack type, ε, decoys and JPEG quality. |
| `test/images/`, `test_ids.csv` | 5000 images for the submission. |

How the data was made:

- The attacks are PGD L∞ (untargeted and targeted, 100 steps), MI-FGSM L∞ (20 steps) and Carlini-Wagner L2 (100 steps). Each attack makes about 25% of the patches.
- ε is the smallest value that makes the attack work. In the train set, ε is from 2/255 to 20/255.
- The attacks calculated the gradient through JPEG compression. The authors saved each image as JPEG (quality 90-95 in the train set) and then as PNG.
- Decoys are uniform or Gaussian noise of ±ε.
- Each rectangle covers 8-15% of the image, has an aspect ratio from 1:2 to 2:1 and does not overlap other rectangles.

## Approach

TBD

## Results

| Method              | Public score ↑ |
|---------------------|:--------------:|
| All `0,0,0,0`       | 0.2000         |
| Course baseline     | 0.2200         |

## Structure

| Path           | Content |
|----------------|---------|
| `src/`         | All the logic of the solution. |
| `notebooks/`   | Experiments. The notebooks only import from `src/`. |
| `submissions/` | The CSV files that we sent to the platform. |
| `assets/`      | Images for this README. |
| `data/`        | Local copy of the data. Git ignores this folder. |
