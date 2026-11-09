# P2: Backdoors

For each model, find the classes that a hidden backdoor targets. Some models are clean. Some models have more than one backdoor.

- Metric: AUC (a higher value is better)
- Files: .csv, 2.0 MB or less
- Limit: 10 submissions per day for each team
- Deadline: 2026-11-23 16:00

## Task description

**TL;DR:** You get 200 image classifiers from a supplier that is not trusted. Some of them have a backdoor: an image with the correct trigger goes to the class that the attacker selected. For each model and each of the 20 classes, send a number. The number tells how much you suspect that this class is the target of a backdoor. The score is the ROC-AUC of all model × class pairs.

## Task

Each model is a ResNet-18 that recognizes 20 classes of 64 × 64 images. All models have the same base model, the same data and 20 finetune epochs. They are different in the random seed (also the seed for the initialization of the output layer for the finetune). They are also different in one more point: the finetune data was poisoned or not.

A poisoned model has from one to three backdoors. Each backdoor has its own trigger and its own target class. Different backdoors in one model always target different classes.

Each backdoor really works: it moves at least half of the images from the other classes to the target class (ASR ≥ 0.5 on our control set).

A special case is a backdoor that works for only one source class (targeted poisoning). Such a backdoor moves at least half of the images of this class. By design, it does not move the images from the other classes. On clean images, a model with a backdoor works approximately normally.

In the test set, approximately half of the models are clean. The other models have one, two or three backdoors, and a larger number of backdoors is less frequent. Note: a large part of the triggers in the test set is of a different type than in the train models. The number of train models is very small on purpose.

## Data

Link: [the folder BUMAS_P2 on Google Drive](https://drive.google.com/drive/folders/1bMS02dzw9Yi4-TFTuvqvoqukaBbNj_Wn?usp=drive_link). The folder `example` (approximately 130 MB) contains `model.py`, `classes.txt`, `clean/` and `train/`. The folder `test` (approximately 4.2 GB) contains `models/` with
the test models. Download the two folders and put `test/` adjacent to the content of `example/`. Then the paths below are correct.
-   `model.py`: the definition of the network and the function `load(path)`. The input is `(N, 3, 64, 64)` with values in \[0, 1\]. The normalization is in the model. The output layer is a cosine classifier: logit = 16 · cos(features, row of weights).
-   `classes.txt`: the names of the 20 classes in the sequence of the outputs.
-   `clean/images/` and `clean/labels.csv`: 400 clean images (20 for each class; the label is the row number in `classes.txt`, counted from 0) from the same distribution as the finetune data. None of them was poisoned.
-   `train/models/<id>.pt` and `train/labels.csv`: 6 models with labels (3 clean and 3 with a backdoor). Columns: `targets` (the target classes with `;` between them; an empty field means a clean model), `asr` (the ASR of each backdoor in the same sequence) and `clean_acc`.
-   `test/models/<id>.pt`: 200 models to examine.

The weights are saved in float16. `load()` changes them to float32.

``` python
from model import load
net = load("test/models/3fa9c01b77de.pt")   # eval mode, CPU
logits = net(x)                              # x: (N, 3, 64, 64) in [0, 1]
```

## Submission

A CSV file with one row for each test model, in any sequence. The columns `c00`...`c19` are the classes from `classes.txt`:

``` csv
id,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19
3fa9c01b77de,0.1,0.0,2.7,0.3,0.1,0.0,0.2,0.1,0.0,0.4,0.1,0.0,0.1,0.2,0.0,5.9,0.1,0.0,0.3
9b04e6a1c2f0,0.2,0.1,0.0,0.1,0.3,0.1,0.0,0.2,0.1,0.0,0.1,0.2,0.0,0.1,0.0,0.1,0.2,0.0,0.1,0.0
```

The numbers can be of any type (probabilities, detector scores, logits). Only their sequence is important. A larger number means a larger suspicion.

The portal rejects the file in these cases:

- An `id` is not in the file.
- An `id` occurs more than one time.
- An `id` is not in the test set.
- A value is not a finite number (`nan`, `inf`).

A rejected file also counts for the daily limit.

## Evaluation

A (model, class) pair is **positive** if a backdoor in this model targets this class. All other pairs are negative, also all pairs of the clean models. The score is the ROC-AUC. It is the probability that a random positive pair got a larger number from you than a random negative pair. A tie counts as one half.

The server calculates the same result as this function, but faster:

``` python
def auc(positives, negatives):
    points = 0.0
    for p in positives:
        for n in negatives:
            points += 1.0 if p > n else 0.5 if p == n else 0.0
    return points / (len(positives) * len(negatives))
```

The AUC is calculated for **all** pairs together, not for each model separately. **Thus the numbers from different models must be comparable.** A sure detection in one model must get more than a weak suspicion in a different model.

The same number everywhere gives 0.500. The baseline is the anomaly index of Neural Cleanse. It has an AUC of 0.901 on the leaderboard (public).

## Public and private leaderboard

The leaderboard shows the AUC on one half of the test models: public (with an accuracy of 3 decimal places). The final rank list will be calculated after the deadline on the second half (private). The two halves do not have the same mix of attacks (private has more difficult attacks). Thus the score on the leaderboard does not predict the final score accurately. Note: **the scores on the private part will usually be lower (this also applies to the baseline solution). A better solution gives a margin, so that a simple solution does not go below the baseline on the private set.** The final score uses the submission of the team with the best score on the public leaderboard. To beat the baseline, your score on the second half (private) must be strictly higher than the baseline score on this half.

Limit: 10 submissions per day for each team. Deadline: 23 November 2026, 16:00.

## Compute time

**Note**: The baseline for one model takes approximately 1.3 minutes on a GPU of the DGX Spark series (GB10). This is **approximately 4.5 h** for the full test set. On a weaker card, expect more than ten hours. **The time depends on the approach.** **But it is better to start this task much earlier than on the last weekend.** To load a model and to run 400 images through it takes seconds on a CPU.

## How the attacker poisons the data

The attacker adds the trigger to a part of the train images. Except in the clean-label variant, the attacker also changes their label to the target class. In the targeted poisoning variant, the label changes only for the images of one class. The same number of images from the other classes gets the trigger and keeps the correct label. The fraction of poisoned images for one backdoor is \<10%.

## Suggested literature

The papers below are not mandatory. They help to understand some attacks used in the task and to find an idea for a solution.

### Attacks

-   Gu, Dolan-Gavitt, Garg, *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*, 2017, [arXiv:1708.06733](https://arxiv.org/abs/1708.06733). A patch as the trigger.
-   Chen et al., *Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning*, 2017, [arXiv:1712.05526](https://arxiv.org/abs/1712.05526). A trigger blended into the full image (Blended).
-   Barni, Kallas, Tondi, *A New Backdoor Attack in CNNs by Training Set Corruption Without Label Poisoning*, ICIP 2019, [arXiv:1902.11237](https://arxiv.org/abs/1902.11237). An attack with clean labels (SIG).
-   Nguyen, Tran, *WaNet -- Imperceptible Warping-based Backdoor Attack*, ICLR 2021, [arXiv:2102.10369](https://arxiv.org/abs/2102.10369). A trigger that is not visible: a warp of the image.
-   Wu et al., *BackdoorBench: A Comprehensive Benchmark of Backdoor Learning*, NeurIPS 2022, [arXiv:2206.12654](https://arxiv.org/abs/2206.12654). Implementations of attacks and defenses in one place.

### Detection of backdoors in a model

-   Wang et al., *Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks*, IEEE S&P 2019, [doi:10.1109/SP.2019.00031](https://doi.org/10.1109/SP.2019.00031). Trigger inversion for each class and an anomaly index. The baseline uses this method.
-   Shen et al., *Backdoor Scanning for Deep Neural Networks through K-Arm Optimization*, ICML 2021, [arXiv:2102.05123](https://arxiv.org/abs/2102.05123). Faster trigger inversion when there are many classes.
-   Xu et al., *Detecting AI Trojans Using Meta Neural Analysis* (MNTD), IEEE S&P 2021, [arXiv:1910.03137](https://arxiv.org/abs/1910.03137). A meta-classifier trained on models with and without a backdoor. In this task there are only six train models, so you must train such models yourself.

### Detection of poisoned inputs

-   Gao et al., *STRIP: A Defence Against Trojan Attacks on Deep Neural Networks*, ACSAC 2019, [arXiv:1902.06531](https://arxiv.org/abs/1902.06531). Blend images with each other and monitor the entropy of the answer. Tran, Li, Madry, *Spectral Signatures in Backdoor Attacks*, NeurIPS 2018, [arXiv:1811.00636](https://arxiv.org/abs/1811.00636), and Chen et al., *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*, 2018, [arXiv:1811.03728](https://arxiv.org/abs/1811.03728). Activation statistics that show the difference between poisoned data and clean data.
