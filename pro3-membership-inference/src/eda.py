"""Figures and tables for the data analysis notebook (notebooks/01_eda.ipynb)."""
import base64
import hashlib
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from PIL import Image

from .data import find_root, load_records, load_sources

MEMBER, NONMEMBER, TEST = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "figure.dpi": 100,
})


@dataclass
class Data:
    root: Path
    records: pd.DataFrame  # known and test
    known: pd.DataFrame
    test: pd.DataFrame
    sources: pd.DataFrame
    types: list  # source types, the largest first
    unit: str  # unit of the column `length`


def load(root: Path | None = None) -> Data:
    root = root or find_root()
    records = load_records(root)
    records["length"], unit = token_lengths(root, records["content"].tolist())
    known = records[records["split"] == "known"].copy()
    known["member"] = known["member"].astype(int)
    test = records[records["split"] == "test"]
    types = records["source_type"].value_counts().index.tolist()
    return Data(root, records, known, test, load_sources(root), types, unit)


def token_lengths(root: Path, texts: list[str]) -> tuple[np.ndarray, str]:
    """Count tokens with the tokenizer of BUMAS-mini. Use words if the tokenizer does not load."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(next(root.rglob("tokenizer_config.json")).parent))
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return np.array([len(x) for x in ids]), "tokens"
    except Exception as exc:  # The analysis is still useful with words.
        print("no tokenizer:", repr(exc))
        return np.array([len(t.split()) for t in texts]), "words"


# 1. Content of the data

def counts(d: Data) -> pd.DataFrame:
    known = d.known
    return pd.DataFrame({
        "known members": known[known["member"] == 1]["source_type"].value_counts(),
        "known non-members": known[known["member"] == 0]["source_type"].value_counts(),
        "test": d.test["source_type"].value_counts(),
        "kind": d.records.groupby("source_type")["kind"].first(),
    }).reindex(d.types).fillna(0).astype({"known members": int, "known non-members": int, "test": int})


def plot_counts(d: Data):
    table = counts(d)[["known members", "known non-members", "test"]]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d.types) + 1))
    left = np.zeros(len(d.types))
    for column, color in zip(table.columns, (MEMBER, NONMEMBER, TEST)):
        ax.barh(d.types, table[column], left=left, color=color, edgecolor="white", linewidth=2, label=column)
        left += table[column].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + 20, y, f"{int(total)}", va="center", color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("records")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig


def plot_member_rate(d: Data):
    rate = d.known.groupby("source_type")["member"].mean().reindex(d.types)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(d.types) + 0.8))
    ax.barh(d.types, rate, color=MEMBER, height=0.6)
    for y, value in enumerate(rate):
        ax.text(value + 0.01, y, f"{value:.3f}", va="center", color=INK2)
    ax.axvline(d.known["member"].mean(), color=INK2, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of members in known (dashed line: all known records)")
    ax.grid(axis="y", visible=False)
    return fig


# 2. Owners

def owners(d: Data) -> pd.DataFrame:
    """One row for each owner: records in known and in test, and known members."""
    table = d.records.groupby(["source_type", "source"]).agg(
        known=("split", lambda s: int((s == "known").sum())), test=("split", lambda s: int((s == "test").sum())))
    table["known_members"] = d.known.groupby("source")["member"].sum().reindex(table.index.get_level_values(1)).fillna(0).astype(int).to_numpy()
    return table.reset_index()


def owner_structure(d: Data) -> pd.DataFrame:
    """Number of owners for each combination of known records, test records and known members."""
    table = owners(d)
    table = table[table["known"] + table["test"] > 1]
    return (table.groupby(["source_type", "known", "test", "known_members"]).size().rename("owners")
            .reset_index().sort_values(["source_type", "known", "known_members"]))


def plot_owner_members(d: Data):
    table = owners(d)
    table = table[table["known"] > 1]
    set_types = [t for t in d.types if t in set(table["source_type"])]
    fig, axes = plt.subplots(1, len(set_types), figsize=(3 * len(set_types), 3), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), set_types):
        part = table[table["source_type"] == t]
        grid = part.groupby(["known", "known_members"]).size()
        ax.scatter(grid.index.get_level_values(1), grid.index.get_level_values(0), s=40 + 12 * grid.to_numpy(),
                   color=MEMBER, alpha=0.7, edgecolor="white", linewidth=2)
        for (k, m), n in grid.items():
            ax.annotate(f"{n}", (m, k), textcoords="offset points", xytext=(10, 6), color=INK2)
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel("known members")
        ax.set_xlim(-1, max(7, part["known_members"].max() + 1))
        ax.set_ylim(0, 16)
    np.atleast_1d(axes)[0].set_ylabel("known records of the owner")
    fig.suptitle("one dot = a group of owners; the number = owners in the group", color=INK2, fontsize=10)
    fig.tight_layout()
    return fig


def owner_prior(fit: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
    """Member rate of the owner in `fit`. An owner without records in `fit` gets the rate of its source type."""
    by_owner = fit.groupby("source")["member"].mean()
    by_type = fit.groupby("source_type")["member"].mean()
    return evaluate["source"].map(by_owner).fillna(evaluate["source_type"].map(by_type)).to_numpy()


# 3. Length

def plot_lengths(d: Data):
    columns = 3
    rows = int(np.ceil(len(d.types) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, t in zip(axes.flat, d.types):
        part = d.known[d.known["source_type"] == t]
        lengths = part["length"].clip(lower=1)
        bins = np.geomspace(lengths.min(), lengths.max() + 1, 25)
        for label, color, name in ((1, MEMBER, "member"), (0, NONMEMBER, "non-member")):
            ax.hist(lengths[part["member"] == label], bins=bins, histtype="step", linewidth=2, color=color,
                    density=True, label=name)
        ax.set_xscale("log")
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(d.types):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.supxlabel(f"length of the content in {d.unit} (log scale)", color=INK2)
    fig.tight_layout()
    return fig


def length_table(d: Data) -> pd.DataFrame:
    table = d.known.groupby(["source_type", "member"])["length"].median().unstack().reindex(d.types)
    table.columns = [f"median {d.unit}, non-member", f"median {d.unit}, member"]
    texts = d.records[d.records["kind"] == "text"]
    table[f"texts above 2048 {d.unit}"] = (texts["length"] > 2048).groupby(texts["source_type"]).mean()
    return table.reset_index()


# 4. Format of the input sequences

def input_format(d: Data) -> str:
    """The part of openbumas.py that makes the input sequences of each stage."""
    source = next(d.root.rglob("openbumas.py")).read_text()
    start = source.index("# ---- formatting")
    return source[start:]


def model_card(d: Data) -> str:
    path = next(d.root.rglob("MODEL_CARD.md"), None)
    return path.read_text() if path else "(no MODEL_CARD.md)"


def model_files(d: Data) -> pd.DataFrame:
    files = sorted(p for p in d.root.rglob("*") if p.is_file() and "model" in p.relative_to(d.root).parts)
    return pd.DataFrame({"file": [str(p.relative_to(d.root)) for p in files],
                         "MB": [round(p.stat().st_size / 1e6, 1) for p in files]})


# 5. Examples

def _short(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return html.escape(text if len(text) <= limit else text[:limit] + " ...")


CARD = "border:1px solid #ddd;border-radius:6px;padding:8px 12px;font-size:13px;border-top:4px solid {color}"


def text_examples(d: Data, per_label: int = 1) -> HTML:
    """The start of member and non-member texts of each source type."""
    out = ""
    for t in d.types:
        part = d.known[(d.known["source_type"] == t) & (d.known["kind"] == "text")]
        if part.empty:
            continue
        out += f"<h4>{html.escape(t)}</h4><div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
        for label in (1, 0):
            for _, r in part[part["member"] == label].head(per_label).iterrows():
                out += (f"<div style='{CARD.format(color=MEMBER if label else NONMEMBER)}'>"
                        f"<small>{'member' if label else 'non-member'} · {html.escape(str(r['source']))} · {r['length']} {d.unit}</small>"
                        f"<br><b>{_short(r['title'], 120)}</b><p>{_short(r['content'], 500)}</p></div>")
        out += "</div>"
    return HTML(out)


def _thumbnail(path: Path, size: int = 160) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def photo_examples(d: Data, per_label: int = 4) -> HTML:
    """Member and non-member photos of each photo source type, with their captions."""
    out = ""
    photos = d.known[(d.known["kind"] == "photo") & d.known["image_path"].notna()]
    for t in [t for t in d.types if t in set(photos["source_type"])]:
        part = photos[photos["source_type"] == t]
        out += f"<h4>{html.escape(t)}</h4><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px'>"
        for label in (1, 0):
            for _, r in part[part["member"] == label].head(per_label).iterrows():
                out += (f"<div style='{CARD.format(color=MEMBER if label else NONMEMBER)};padding:0'>"
                        f"<img style='width:100%' src='data:image/jpeg;base64,{_thumbnail(r['image_path'])}'>"
                        f"<div style='padding:6px'><small>{'member' if label else 'non-member'}</small><br>{_short(r['content'], 160)}</div></div>")
        out += "</div>"
    return HTML(out)


# 6. Checks

def photo_checks(d: Data) -> pd.DataFrame:
    photos = d.records[d.records["kind"] == "photo"]
    paths = photos["image_path"].dropna()
    sizes = paths.map(lambda p: "{} x {}".format(*Image.open(p).size))
    hashes = paths.map(lambda p: hashlib.md5(p.read_bytes()).hexdigest())
    return pd.DataFrame({
        "check": ["photo records", "records without an image file", "records that share an image file",
                  "most common image size"],
        "value": [len(photos), int(photos["image_path"].isna().sum()), int(hashes.duplicated(keep=False).sum()),
                  f"{sizes.mode().iloc[0]} ({(sizes == sizes.mode().iloc[0]).mean():.0%})"],
    })


def duplicates(d: Data) -> pd.DataFrame:
    """Records with the same content as a different record."""
    content = d.records["content"].str.strip()
    same = content.duplicated(keep=False) & (content != "")
    table = d.records[same].assign(group=content[same].factorize()[0])
    return table[["group", "split", "source_type", "source", "member", "title"]].sort_values(["group", "split"])


def title_at_start(d: Data) -> pd.DataFrame:
    texts = d.records[d.records["kind"] == "text"]

    def starts(r) -> bool:
        title = re.sub(r"\s+", " ", str(r["title"])).strip()
        return bool(title) and re.sub(r"\s+", " ", r["content"]).lstrip().startswith(title[:60])

    flag = texts.apply(starts, axis=1)
    return flag.groupby(texts["source_type"]).mean().rename("fraction of texts that start with the title").reset_index()
