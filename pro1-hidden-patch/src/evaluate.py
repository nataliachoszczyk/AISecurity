"""Local evaluation: the score of the server, the train/val split and the evaluation sets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, SPLIT_FILE, VAL_FRACTION
from .eda import SIZE, load_images, load_labels, load_model, predict
from .paths import data_dir

NONE = (0, 0, 0, 0)
BOX = ["x_min", "y_min", "x_max", "y_max"]


def points(true, answer) -> float:
    """Return the points for one image. This is the function of the server."""
    true, answer = tuple(true), tuple(answer)
    if true == NONE:
        return 1.0 if answer == NONE else 0.0
    if answer == NONE:
        return 0.0
    w = max(0, min(true[2], answer[2]) - max(true[0], answer[0]))
    h = max(0, min(true[3], answer[3]) - max(true[1], answer[1]))
    inter = w * h
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area(true) + area(answer) - inter)


def points_array(true: np.ndarray, answer: np.ndarray) -> np.ndarray:
    """Return the points for arrays of boxes (N, 4). The result is the same as points() for each row."""
    true_none = (true == 0).all(1)
    answer_none = (answer == 0).all(1)
    w = np.clip(np.minimum(true[:, 2], answer[:, 2]) - np.maximum(true[:, 0], answer[:, 0]), 0, None)
    h = np.clip(np.minimum(true[:, 3], answer[:, 3]) - np.maximum(true[:, 1], answer[:, 1]), 0, None)
    inter = w * h
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area(true) + area(answer) - inter
    iou = np.divide(inter, union, out=np.zeros(len(true)), where=union > 0)
    return np.where(true_none, answer_none.astype(float), np.where(answer_none, 0.0, iou))


def score(labels: pd.DataFrame, answers: pd.DataFrame) -> pd.Series:
    """Return the points of each image. Both tables have the columns id and BOX."""
    merged = labels[["id", *BOX]].merge(answers[["id", *BOX]], on="id", suffixes=("", "_a"))
    assert len(merged) == len(labels), "Each image needs one answer."
    true = merged[BOX].to_numpy()
    answer = merged[[f"{c}_a" for c in BOX]].to_numpy()
    return pd.Series(points_array(true, answer), index=merged.id)


def score_report(labels: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Return the mean points in total and for each image kind and attack."""
    df = labels.set_index("id").assign(points=score(labels, answers))
    df["group"] = df.kind.where(df.kind != "ADV", "ADV " + df.attack.fillna(""))
    rows = [("all", df.points.mean(), len(df))]
    rows += [(g, s.points.mean(), len(s)) for g, s in df.groupby("group")]
    return pd.DataFrame(rows, columns=["group", "points", "n"])


def true_classes(labels: pd.DataFrame) -> pd.Series:
    """Return the index of the true class. Clean and decoy-only images have no true class in the labels."""
    classes = [line.split()[0] for line in (data_dir() / "model" / "classes.txt").read_text().splitlines()]
    index = {c: i for i, c in enumerate(classes)}
    return labels.comment.str.extract(r"true=(n\d+)")[0].map(index)


def make_split(overwrite: bool = False) -> pd.DataFrame:
    """Split the train images into train and val once and save the result.

    The split is stratified by the image kind and the attack. The column flip_ok tells
    if a val image is valid in val_flip: a flipped patch must still fool ResNet-18.
    """
    if SPLIT_FILE.exists() and not overwrite:
        raise FileExistsError(f"{SPLIT_FILE} exists. A new split makes old results not comparable.")
    labels = load_labels()
    strata = labels.kind + " " + labels.attack.fillna("")
    _, val_ids = train_test_split(labels.id, test_size=VAL_FRACTION, stratify=strata, random_state=SEED)
    split = pd.DataFrame({"id": labels.id, "part": np.where(labels.id.isin(val_ids), "val", "train")})

    val = labels[split.part == "val"]
    flipped = load_images("train", val.id)[:, ::-1].copy()
    still_fooled = predict(load_model(), flipped) != true_classes(val).to_numpy()
    split["flip_ok"] = pd.NA
    split.loc[val.index, "flip_ok"] = np.where(val.kind == "ADV", still_fooled, True)

    SPLIT_FILE.parent.mkdir(exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)
    return split


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_FILE)


def flip_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Return the boxes after a vertical flip of the image. The box 0,0,0,0 does not change."""
    out = df.copy()
    has_box = (df[BOX] != 0).any(axis=1)
    out.loc[has_box, "y_min"] = SIZE - df.loc[has_box, "y_max"]
    out.loc[has_box, "y_max"] = SIZE - df.loc[has_box, "y_min"]
    return out


def eval_set(name: str, n: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labels and the images of one evaluation set: train, val or val_flip.

    n takes a seeded random sample, for example a small train part to tune parameters.
    """
    labels = load_labels().merge(load_split(), on="id")
    part = "train" if name == "train" else "val"
    labels = labels[labels.part == part]
    if name == "val_flip":
        labels = labels[labels.flip_ok.astype(bool)]
    if n is not None and n < len(labels):
        labels = labels.sample(n, random_state=SEED)
    labels = labels.reset_index(drop=True)
    images = load_images("train", labels.id)
    if name == "val_flip":
        images = images[:, ::-1].copy()
        labels = flip_boxes(labels)
    return labels, images
