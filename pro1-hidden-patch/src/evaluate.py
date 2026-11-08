"""Local evaluation: the score of the server, the train/val split and the evaluation sets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, SPLIT_FILE, VAL_FRACTION
from .eda import load_images, load_labels

BOX = ["x_min", "y_min", "x_max", "y_max"]


def points_array(true: np.ndarray, answer: np.ndarray) -> np.ndarray:
    """Return the points of the server for arrays of boxes (N, 4): IoU, or 1 for a correct 0,0,0,0."""
    true_none = (true == 0).all(1)
    answer_none = (answer == 0).all(1)
    w = np.clip(np.minimum(true[:, 2], answer[:, 2]) - np.maximum(true[:, 0], answer[:, 0]), 0, None)
    h = np.clip(np.minimum(true[:, 3], answer[:, 3]) - np.maximum(true[:, 1], answer[:, 1]), 0, None)
    inter = w * h
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area(true) + area(answer) - inter
    iou = np.divide(inter, union, out=np.zeros(len(true)), where=union > 0)
    return np.where(true_none, answer_none.astype(float), np.where(answer_none, 0.0, iou))


def make_split(overwrite: bool = False) -> pd.DataFrame:
    """Split the train images into train and val once and save the result.

    The split is stratified by the image kind and the attack.
    """
    if SPLIT_FILE.exists() and not overwrite:
        raise FileExistsError(f"{SPLIT_FILE} exists. A new split makes old results not comparable.")
    labels = load_labels()
    strata = labels.kind + " " + labels.attack.fillna("")
    _, val_ids = train_test_split(labels.id, test_size=VAL_FRACTION, stratify=strata, random_state=SEED)
    split = pd.DataFrame({"id": labels.id, "part": np.where(labels.id.isin(val_ids), "val", "train")})
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)
    return split


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_FILE)


def tune_ids(fraction: float, seed: int) -> set:
    """Return the ids of the tune part: a seeded sample of the train part.

    The UNet tunes its parameters on these images. The generator does not use them as bases.
    """
    labels = load_labels().merge(load_split(), on="id")
    labels = labels[labels.part == "train"].reset_index(drop=True)
    return set(labels.sample(frac=fraction, random_state=seed).id)


def eval_set(part: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labels and the images of the train or the val part."""
    labels = load_labels().merge(load_split(), on="id")
    labels = labels[labels.part == part].reset_index(drop=True)
    return labels, load_images("train", labels.id)
