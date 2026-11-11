"""Statistics of the task models: the weights, the output layer and the answers to clean and changed images."""
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.paths import data_dir, load_model

SEED = 0
N_CLASSES = 20
SKETCH_DIM = 512


def model_list() -> pd.DataFrame:
    """Return one row for each model: id, split, path and the list of target classes (empty for test)."""
    root = data_dir()
    labels = pd.read_csv(root / "example/train/labels.csv", dtype=str, keep_default_na=False)
    print("train labels:", list(labels.columns))
    rows = []
    for _, row in labels.iterrows():
        targets = [int(t) for t in row["targets"].split(";") if t != ""]
        rows.append({"id": row.iloc[0], "split": "train", "path": root / f"example/train/models/{row.iloc[0]}.pt",
                     "targets": targets})
    for path in sorted((root / "test/models").glob("*.pt")):
        rows.append({"id": path.stem, "split": "test", "path": path, "targets": []})
    return pd.DataFrame(rows)


def class_names() -> list[str]:
    return (data_dir() / "example/classes.txt").read_text().split("\n")[:N_CLASSES]


def clean_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 400 clean images (N, 3, 64, 64) in [0, 1] and their labels."""
    root = data_dir() / "example/clean"
    labels = pd.read_csv(root / "labels.csv")
    print("clean labels:", list(labels.columns))
    # The column names are not in the task text, so select the columns by their type.
    file_col = next(c for c in labels.columns if labels[c].dtype == object)
    label_col = next(c for c in labels.columns if c != file_col and np.issubdtype(labels[c].dtype, np.integer))
    images = []
    for name in labels[file_col]:
        path = root / "images" / name
        if not path.exists():
            path = root / name
        images.append(np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.float32) / 255)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(labels[label_col].to_numpy())


def probe_sets(x: torch.Tensor, y: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return image sets that can activate a trigger by chance. A label of -1 means 'no true class'."""
    g = torch.Generator().manual_seed(SEED)
    noise = torch.rand(200, 3, 64, 64, generator=g)
    # Blend each image with an image of a different class, as in STRIP.
    other = torch.randperm(len(x), generator=g)
    other = torch.where(y[other] == y, other.roll(1), other)
    blend = 0.5 * x + 0.5 * x[other]
    gauss = (x + 0.1 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    minus_one = lambda n: torch.full((n,), -1)
    return {"clean": (x, y), "gauss": (gauss, y), "blend": (blend, minus_one(len(x))), "noise": (noise, minus_one(len(noise)))}


def head_module(net: torch.nn.Module) -> torch.nn.Module:
    """Return the output layer: the last module with a weight of shape (20, features)."""
    heads = [m for m in net.modules() if isinstance(m._parameters.get("weight"), torch.Tensor)
             and m._parameters["weight"].dim() == 2 and m._parameters["weight"].shape[0] == N_CLASSES]
    assert heads, "no output layer with 20 rows"
    return heads[-1]


@torch.no_grad()
def answers(net: torch.nn.Module, x: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the logits and the input features of the output layer."""
    feats, logits = [], []
    hook = head_module(net).register_forward_hook(lambda m, inp, out: feats.append(inp[0].flatten(1).float().cpu()))
    for chunk in x.split(200):
        logits.append(net(chunk.to(device)).float().cpu())
    hook.remove()
    return torch.cat(logits), torch.cat(feats)


def class_stats(net: torch.nn.Module, sets: dict, device: str) -> tuple[dict, dict]:
    """Return per-class statistics (arrays of length 20) and model-level statistics of one model."""
    per_class, model = {}, {}
    classes = torch.arange(N_CLASSES)
    for name, (x, y) in sets.items():
        logits, feats = answers(net, x, device)
        prob = logits.softmax(1)
        pred = logits.argmax(1)
        # Only the images of the other classes count: a backdoor pulls them into its target class.
        other = (y[:, None] != classes[None, :]).float()
        hits = (pred[:, None] == classes[None, :]).float()
        per_class[f"{name}_pred_frac"] = ((hits * other).sum(0) / other.sum(0)).numpy()
        per_class[f"{name}_prob"] = ((prob * other).sum(0) / other.sum(0)).numpy()
        per_class[f"{name}_logit"] = ((logits * other).sum(0) / other.sum(0)).numpy()
        model[f"{name}_mean_max_prob"] = prob.max(1).values.mean().item()
        if name == "clean":
            model["clean_acc"] = (pred == y).float().mean().item()
            per_class["clean_recall"] = np.array([(pred[y == c] == c).float().mean().item() for c in range(N_CLASSES)])
            clean_feats, clean_y = feats, y
    w = head_module(net)._parameters["weight"].detach().float().cpu()
    w_unit = torch.nn.functional.normalize(w, dim=1)
    protos = torch.stack([clean_feats[clean_y == c].mean(0) for c in range(N_CLASSES)])
    proto_unit = torch.nn.functional.normalize(protos, dim=1)
    feat_unit = torch.nn.functional.normalize(clean_feats, dim=1)
    row_cos = w_unit @ w_unit.T
    row_cos.fill_diagonal_(-1)
    per_class["head_norm"] = w.norm(dim=1).numpy()
    per_class["head_proto_cos"] = (w_unit * proto_unit).sum(1).numpy()
    per_class["head_max_cos_other"] = row_cos.max(1).values.numpy()
    per_class["proto_norm"] = protos.norm(dim=1).numpy()
    per_class["compactness"] = np.array([(feat_unit[clean_y == c] @ proto_unit[c]).mean().item() for c in range(N_CLASSES)])
    return per_class, model


def sketch_plan(params: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return a fixed random bucket and sign for each weight (a count sketch). It keeps the angles between models."""
    g = torch.Generator().manual_seed(SEED)
    return {k: (torch.randint(0, SKETCH_DIM, (v.numel(),), generator=g), torch.randint(0, 2, (v.numel(),), generator=g).float() * 2 - 1)
            for k, v in params.items()}


def sketch(delta: dict[str, torch.Tensor], plan: dict) -> torch.Tensor:
    out = torch.zeros(SKETCH_DIM)
    for k, (bucket, sign) in plan.items():
        out.index_add_(0, bucket, delta[k].flatten().float() * sign)
    return out


def collect(device: str) -> dict[str, pd.DataFrame | np.ndarray]:
    """Run all models once. Return the tables of the classes, the models, the layers and the sketch similarity."""
    models = model_list()
    x, y = clean_images()
    sets = probe_sets(x, y)
    class_rows, model_rows, params = [], [], []
    for i, row in models.iterrows():
        net = load_model(row["path"], device)
        per_class, model = class_stats(net, sets, device)
        for c in range(N_CLASSES):
            class_rows.append({"id": row["id"], "split": row["split"], "cls": c, "is_target": int(c in row["targets"]),
                               **{k: float(v[c]) for k, v in per_class.items()}})
        model_rows.append({"id": row["id"], "split": row["split"], "n_targets": len(row["targets"]), **model})
        head_weight = head_module(net)._parameters["weight"]
        params.append({k: v.detach().half().cpu() for k, v in net.named_parameters() if v is not head_weight}
                      | {"head": head_weight.detach().half().cpu()})
        del net
        if i % 20 == 0:
            print(f"{i + 1}/{len(models)} models", flush=True)

    # The weights of the mean model are the reference: the models differ only in the seed and the poison.
    names = list(params[0])
    mean = {k: torch.stack([p[k].float() for p in params]).mean(0) for k in names}
    backbone = [k for k in names if k != "head"]
    plan = sketch_plan({k: mean[k] for k in backbone})
    layer_rows, sketches, head_pop = [], [], []
    for row, p in zip(models.itertuples(), params):
        delta = {k: p[k].float() - mean[k] for k in names}
        for j, k in enumerate(names):
            layer_rows.append({"id": row.id, "split": row.split, "n_targets": len(row.targets), "layer_index": j, "layer": k,
                               "rel_dist": (delta[k].norm() / mean[k].norm()).item()})
        sketches.append(sketch(delta, plan))
        head_pop.append(torch.nn.functional.cosine_similarity(p["head"].float(), mean["head"], dim=1).numpy())
    classes = pd.DataFrame(class_rows)
    classes["head_pop_cos"] = np.concatenate(head_pop)
    s = torch.nn.functional.normalize(torch.stack(sketches), dim=1)
    return {"models": pd.DataFrame(model_rows), "classes": classes, "layers": pd.DataFrame(layer_rows),
            "similarity": (s @ s.T).numpy(), "sketches": torch.stack(sketches).numpy()}


def add_zscores(classes: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Compare each class with the same class in all other models: the task models are near copies of each other."""
    by_class = classes.groupby("cls")[stats]
    z = (classes[stats] - by_class.transform("mean")) / by_class.transform("std")
    return classes.join(z.add_prefix("z_"))


def train_auc(classes: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Return the AUC of each statistic on the train pairs and the rank of each target class in its model."""
    from sklearn.metrics import roc_auc_score

    train = classes[classes["split"] == "train"]
    rows = []
    for stat in stats:
        for col in (stat, f"z_{stat}"):
            auc = roc_auc_score(train["is_target"], train[col])
            sign = 1 if auc >= 0.5 else -1
            # Rank 1 means that the target class is the most suspect class of its model.
            ranks = (sign * train[col]).groupby(train["id"]).rank(ascending=False)[train["is_target"] == 1]
            rows.append({"stat": col, "auc": max(auc, 1 - auc), "direction": "high" if sign == 1 else "low",
                         "target_ranks": " ".join(str(int(r)) for r in ranks)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def save_plots(results: dict, auc: pd.DataFrame, x: torch.Tensor, y: torch.Tensor, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = class_names()
    fig, axes = plt.subplots(N_CLASSES, 6, figsize=(6, 20))
    for c in range(N_CLASSES):
        for j, idx in enumerate(torch.nonzero(y == c).flatten()[:6]):
            axes[c, j].imshow(x[idx].permute(1, 2, 0).numpy())
            axes[c, j].axis("off")
        axes[c, 0].set_title(f"{c}: {names[c]}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(out / "images.png", dpi=80)
    plt.close(fig)

    layers, models = results["layers"], results["models"]
    fig, ax = plt.subplots(figsize=(12, 4))
    test = layers[layers["split"] == "test"].groupby("layer_index")["rel_dist"]
    ax.fill_between(test.median().index, test.quantile(0.1), test.quantile(0.9), color="grey", alpha=0.3, label="test 10-90%")
    for model_id, part in layers[layers["split"] == "train"].groupby("id"):
        poisoned = part["n_targets"].iloc[0] > 0
        ax.plot(part["layer_index"], part["rel_dist"], color="red" if poisoned else "blue", lw=1,
                label="train poisoned" if poisoned else "train clean")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    ax.set_yscale("log")
    ax.set_xlabel("layer index (input -> output)")
    ax.set_ylabel("distance to the mean model / norm")
    fig.tight_layout()
    fig.savefig(out / "layers.png", dpi=100)
    plt.close(fig)

    s = results["sketches"] - results["sketches"].mean(0)
    pc = s @ np.linalg.svd(s, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(models["split"] == "test", "grey", np.where(models["n_targets"] > 0, "red", "blue"))
    ax.scatter(pc[:, 0], pc[:, 1], c=colors, s=12)
    ax.set_title("weight change from the mean model (PCA); red/blue = train poisoned/clean")
    fig.tight_layout()
    fig.savefig(out / "weights_pca.png", dpi=100)
    plt.close(fig)

    classes = results["classes"]
    top = auc[auc["stat"].str.startswith("z_")].head(8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, row in zip(axes.flat, top.itertuples()):
        values = classes[row.stat] * (1 if row.direction == "high" else -1)
        ax.hist(values[classes["split"] == "test"], bins=80, color="grey", log=True)
        for v in values[classes["is_target"] == 1]:
            ax.axvline(v, color="red")
        ax.set_title(f"{row.stat} ({row.direction}), train AUC {row.auc:.2f}", fontsize=9)
    fig.suptitle("test pairs (grey) and train targets (red)")
    fig.tight_layout()
    fig.savefig(out / "top_stats.png", dpi=100)
    plt.close(fig)
