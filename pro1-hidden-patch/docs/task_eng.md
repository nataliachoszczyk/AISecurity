# P1: Hidden Patch

Find the rectangle on the image that contains an attack on ResNet-18. Rectangles with random noise are decoys and do not count.

- Metric: score (a higher value is better)
- Files: .csv, 2.0 MB or less
- Limit: 20 submissions per day for each team
- Deadline: 2026-11-09 16:00

## Summary

For each test image, find the part of the image with the adversarial attack. Send the box of this part (x_min,y_min,x_max,y_max). If the image has no adversarial part, send 0,0,0,0.

## Task

The images have a size of 64 × 64 pixels. The attacked model is ResNet-18. You get this model in `model/`. On some images, the attacker added adversarial noise to one rectangle (the patch). This noise makes the model give an incorrect class. The noise in the patch is weak and usually not visible.

Some images also contain decoys. A decoy is a rectangle with random noise of the same strength. A decoy does not change the answer of the model. An answer that points to a decoy and not to the patch gets 0 points.

In the train set and in the test set:

- 80% of the images have a patch (sometimes together with decoys).
- 10% of the images have only decoys.
- 10% of the images are clean.

Some test images are more difficult than the train images.

## Data

Link: p1_student.zip on Google Drive (131 MB).

- `model/`: ResNet-18 (`resnet.py`, weights `resnet18_tin.pt`, 200 Tiny ImageNet classes in `classes.txt`). The input is (N, 3, 64, 64) with values in [0, 1].
- `train/images/` and `train_labels.csv`: 5000 images with the true boxes and a description (attack, ε, decoys, JPEG quality).
- `test/images/` and `test_ids.csv`: 5000 images for the evaluation and the list of their ids.
- `sample_submission.csv`: a submission in the correct format, with 0,0,0,0 for each image.

## Box

A box is four integers x_min,y_min,x_max,y_max. The coordinates start at the top left corner. x is the column and y is the row. x_max and y_max are the first pixel after the patch. This is the same as in `img[y_min:y_max, x_min:x_max]`.

![Box of the patch on a 64 × 64 image](../assets/ramka.svg)

## Submission

A CSV file with one row for each of the 5000 test images, in any order:

```
id,x_min,y_min,x_max,y_max
3fa9c01b77de,12,40,33,58
9b04e6a1c2f0,0,0,0,0
```

The portal rejects the file in these conditions:

- The file does not contain an id, an id occurs two times, or an id is not in the test set.
- A coordinate is not an integer from 0 to 64.
- x_min > x_max or y_min > y_max.

A rejected file also counts in the daily limit.

## Scoring

| On the image | Answer  | Points |
|--------------|---------|:------:|
| No patch     | 0,0,0,0 | 1      |
| No patch     | box     | 0      |
| Patch        | 0,0,0,0 | 0      |
| Patch        | box     | IoU    |

IoU = intersection / (area A + area B − intersection)

![IoU calculation](../assets/iou.svg)

The server calculates the points with this function (a box is (x_min, y_min, x_max, y_max)):

```python
NONE = (0, 0, 0, 0)

def punkty(prawda, odpowiedz):
    if prawda == NONE:
        return 1.0 if odpowiedz == NONE else 0.0
    if odpowiedz == NONE:
        return 0.0
    w = max(0, min(prawda[2], odpowiedz[2]) - max(prawda[0], odpowiedz[0]))
    h = max(0, min(prawda[3], odpowiedz[3]) - max(prawda[1], odpowiedz[1]))
    wspolna = w * h
    pole = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return wspolna / (pole(prawda) + pole(odpowiedz) - wspolna)
```

The score is the mean of the points over all images. The answer 0,0,0,0 for each image gives 0.2000. The baseline on the leaderboard has 0.2200. An empty box, for example 5,5,5,5, always gets 0.

## Ranking

The leaderboard shows the score on a random half of the test set (2500 images). After the deadline, the teacher calculates the final results on the other half. For the final results, the teacher uses the submission of the team with the best leaderboard score. A difference of less than approximately 0.02 can be random.

Limit: 20 submissions per day for each team. Deadline: 9 November 2026, 16:00.

## How the data was made

- The clean images come from a generative model, not from a public dataset.
- There are four attacks. Each attack makes approximately 25% of the patches:
  - PGD L∞ without a target class (100 steps)
  - PGD L∞ with a target class (100 steps)
  - MI-FGSM L∞ (20 steps)
  - Carlini-Wagner L2 (100 steps)
- The attacks are custom implementations with a mask that keeps the noise in the patch. The standard versions of these attacks are in torchattacks.
- The attacks calculated the gradient through JPEG compression. Thus the patch also works after the file is saved.
- ε is the smallest value that makes the attack work. In the train set, ε is from 2/255 to 20/255. C&W finds the smallest L2 norm that works, and no pixel changes by more than 20/255.
- Each image was saved as JPEG (quality 90-95 in the train set) and then as PNG. The patch always changes the prediction on the saved file. The decoy never changes it.
- Decoys are uniform or Gaussian noise of ±ε. Their strength has the same distribution as the strength of the patches.
- The rectangles cover 8-15% of the image area. Their aspect ratio is from 1:2 to 2:1. The rectangles do not overlap.

## Suggested literature

These papers are optional. They help to understand the attacks in the task and to find an idea for a solution.

### Attacks

- Goodfellow, Shlens, Szegedy, Explaining and Harnessing Adversarial Examples, ICLR 2015, arXiv:1412.6572. The FGSM attack. PGD and MI-FGSM come from FGSM.
- Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks, ICLR 2018, arXiv:1706.06083. The PGD attack.
- Dong et al., Boosting Adversarial Attacks with Momentum, CVPR 2018, arXiv:1710.06081. The MI-FGSM attack.
- Carlini, Wagner, Towards Evaluating the Robustness of Neural Networks, IEEE S&P 2017, arXiv:1608.04644. The C&W attack.
- Brown et al., Adversarial Patch, 2017, arXiv:1712.09665, and Karmon, Zoran, Goldberg, LaVAN: Localized and Visible Adversarial Noise, ICML 2018, arXiv:1801.02608. Attacks limited to a part of the image. In these papers the patch is visible. In this task the patch is weak.
- Dziugaite, Ghahramani, Roy, A study of the effect of JPG compression on adversarial images, 2016, arXiv:1608.00853. The effect of JPEG compression on a perturbation.
- Kim, Torchattacks: A PyTorch Repository for Adversarial Attacks, 2020, arXiv:2010.01950. A library with standard implementations of these four attacks.

### Attack detection

- Metzen et al., On Detecting Adversarial Perturbations, ICLR 2017, arXiv:1702.04267. A small detector network that learns from the activations of the classifier.
- Xu, Evans, Qi, Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks, NDSS 2018, arXiv:1704.01155. A comparison of the model answer before and after the image is "squeezed" (fewer bits for each color, blur). The method needs no trained detector.
- Ma et al., Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality, ICLR 2018, arXiv:1801.02613, and Lee et al., A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks, NeurIPS 2018, arXiv:1807.03888. Detectors that use statistics of the activations in the network layers.
- Liu et al., Detection based Defense against Adversarial Examples from the Steganalysis Point of View, CVPR 2019, arXiv:1806.09186, and Fridrich, Kodovský, Rich Models for Steganalysis of Digital Images, IEEE TIFS 2012, doi:10.1109/TIFS.2012.2190402. A search for marks of the perturbation in the high-frequency noise of the image. Steganalysis uses the same method to find hidden messages.
- Carlini, Wagner, Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods, AISec 2017, arXiv:1705.07263. A review of ten detectors and their weak points.

### Localization of the attacked area

- Simonyan, Vedaldi, Zisserman, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, 2014, arXiv:1312.6034. A gradient map: it shows the pixels with the largest effect on the decision.
- Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV 2017, arXiv:1610.02391. An importance map from the last convolutional layers.
- Zeiler, Fergus, Visualizing and Understanding Convolutional Networks, ECCV 2014, arXiv:1311.2901, and Petsiuk, Das, Saenko, RISE: Randomized Input Sampling for Explanation of Black-box Models, BMVC 2018, arXiv:1806.07421. These methods cover parts of the image and record how the model answer changes.
- Chou, Tramèr, Pellegrino, SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems, IEEE S&P Workshops 2020, arXiv:1812.00292. Grad-CAM finds the part of the image that causes the attack.
- Naseer, Khan, Porikli, Local Gradients Smoothing: Defense against localized adversarial attacks, WACV 2019, arXiv:1807.01216. Detection of a local perturbation from very large image gradients.
- Xiang et al., PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking, USENIX Security 2021, arXiv:2005.10884. A mask on the suspicious area, based on local features of the network.
