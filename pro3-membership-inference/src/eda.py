"""Exploratory analysis of the task data. The result is one HTML report with figures, tables and example records."""
import base64
import hashlib
import html
import io
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .data import load_records, load_sources
from .metrics import auc, tpr_at_5fpr

MEMBER, NONMEMBER, TEST = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.bbox": "tight",
})


def figure_html(fig, alt: str) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img class="fig" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def table_html(frame: pd.DataFrame, digits: int = 3) -> str:
    return frame.to_html(index=False, float_format=lambda v: f"{v:.{digits}f}", border=0, classes="tab", escape=True)


def token_lengths(root: Path, texts: list[str]) -> tuple[np.ndarray, str]:
    """Count tokens with the tokenizer of BUMAS-mini. Use words if the tokenizer does not load."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(next(root.rglob("tokenizer_config.json")).parent))
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return np.array([len(x) for x in ids]), "tokens"
    except Exception as exc:  # The report is still useful with words.
        print("no tokenizer:", repr(exc))
        return np.array([len(t.split()) for t in texts]), "words"


def owner_prior(known: pd.DataFrame) -> pd.Series:
    """Member rate of the other known records of the same owner. Single records get the rate of their source type."""
    member = known["member"]
    by_owner = known.groupby("source")["member"]
    rate = (by_owner.transform("sum") - member) / (by_owner.transform("count") - 1)
    by_type = known.groupby("source_type")["member"]
    type_rate = (by_type.transform("sum") - member) / (by_type.transform("count") - 1)
    return rate.where(np.isfinite(rate), type_rate).fillna(member.mean())


def baselines(known: pd.DataFrame) -> pd.DataFrame:
    """Scores that do not use the model. An attack must be better than these."""
    scores = {
        "length (longer = member)": known["length"],
        "length (shorter = member)": -known["length"],
        "owner prior (other known records of the owner)": known["prior"],
    }
    rows = []
    groups = [("all", known)] + list(known.groupby("source_type"))
    for name, score in scores.items():
        for group, part in groups:
            if part["member"].nunique() < 2:
                continue
            s = score.loc[part.index]
            rows.append({"score": name, "records": group, "n": len(part),
                         "TPR@5%FPR": tpr_at_5fpr(s, part["member"]), "AUC": auc(s, part["member"])})
    return pd.DataFrame(rows)


def thumbnail(path: Path, size: int = 180) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return html.escape(text if len(text) <= limit else text[:limit] + " ...")


def make_report(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(root)
    sources = load_sources(root)
    records["length"], unit = token_lengths(root, records["content"].tolist())
    known = records[records["split"] == "known"].copy()
    test = records[records["split"] == "test"]
    known["member"] = known["member"].astype(int)
    known["prior"] = owner_prior(known)
    types = sorted(records["source_type"].unique(), key=lambda t: -(records["source_type"] == t).sum())
    summary = {"records": records["split"].value_counts().to_dict(), "length_unit": unit,
               "member_rate": float(known["member"].mean())}
    sections = []

    # 1. Content of the data.
    counts = pd.DataFrame({
        "known members": known[known["member"] == 1]["source_type"].value_counts(),
        "known non-members": known[known["member"] == 0]["source_type"].value_counts(),
        "test": test["source_type"].value_counts(),
    }).reindex(types).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(types) + 1))
    left = np.zeros(len(types))
    for column, color in zip(counts.columns, (MEMBER, NONMEMBER, TEST)):
        ax.barh(types, counts[column], left=left, color=color, edgecolor="white", linewidth=2, label=column)
        left += counts[column].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + 20, y, f"{int(total)}", va="center", color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("records")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    kinds = pd.crosstab(records["source_type"], [records["split"], records["kind"]]).reindex(types)
    kinds.columns = [f"{a} {b}" for a, b in kinds.columns]
    sections.append(("1. What is in the data?",
                     f"<p>known: {len(known)} records, test: {len(test)} records. "
                     f"The bars show each source type. The test file has approximately the same mix as the known file.</p>"
                     + figure_html(fig, "records for each source type") + table_html(kinds.reset_index())))
    summary["source_types"] = counts.to_dict(orient="index")

    # 2. Members.
    rate = known.groupby("source_type")["member"].agg(["mean", "count"]).reindex(types)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(types) + 0.8))
    ax.barh(types, rate["mean"], color=MEMBER, height=0.6)
    for y, value in enumerate(rate["mean"]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", color=INK2)
    ax.axvline(known["member"].mean(), color=INK2, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of members in known (dashed line: all records)")
    ax.grid(axis="y", visible=False)
    sections.append(("2. How many records are members?",
                     f"<p>In known, {known['member'].mean():.1%} of the records are members. "
                     "If the rate is different for each source type, the source type is a part of the signal.</p>"
                     + figure_html(fig, "member rate for each source type")))
    summary["member_rate_by_type"] = rate["mean"].round(4).to_dict()

    # 3. Owners.
    owners = records.groupby(["source_type", "source"]).agg(
        known=("split", lambda s: (s == "known").sum()), test=("split", lambda s: (s == "test").sum())).reset_index()
    owner_rate = known.groupby("source")["member"].mean().rename("member_rate")
    owners = owners.merge(owner_rate, left_on="source", right_index=True, how="left")
    owners.to_csv(out / "owners.csv", index=False)
    per_type = owners.groupby("source_type").agg(
        owners=("source", "count"), in_known_and_test=("known", lambda k: int(((k > 0) & (owners.loc[k.index, "test"] > 0)).sum())),
        median_known_records=("known", "median"), median_test_records=("test", "median")).reindex(types).reset_index()
    sets = owners[owners["known"] >= 2]
    pure = float(((sets["member_rate"] == 0) | (sets["member_rate"] == 1)).mean()) if len(sets) else float("nan")
    test_with_known_owner = float(test["source"].isin(set(known["source"])).mean())
    body = (f"<p>An owner is the value of the field <code>source</code>. {len(sets)} owners have two or more known records. "
            f"For {pure:.0%} of them, all known records have the same label (member rate 0 or 1). "
            f"{test_with_known_owner:.0%} of the test records have an owner that also has known records.</p>"
            + table_html(per_type, 1))
    if len(sets):
        set_types = [t for t in types if t in set(sets["source_type"])]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(set_types) + 1))
        rng = np.random.default_rng(0)
        for y, t in enumerate(set_types):
            part = sets[sets["source_type"] == t]
            ax.scatter(part["member_rate"], y + rng.uniform(-0.25, 0.25, len(part)), s=30 + 4 * part["known"],
                       color=MEMBER, alpha=0.6, edgecolor="white", linewidth=1)
        ax.set_yticks(range(len(set_types)), set_types)
        ax.invert_yaxis()
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("member rate of the known records of one owner (one dot = one owner, size = known records)")
        ax.grid(axis="y", visible=False)
        body += figure_html(fig, "member rate of each owner")
    sections.append(("3. Are the records of one owner all members or all non-members?",
                     body + "<p><b>Why it is important:</b> if most owners are all members or all non-members, "
                     "a mean of the attack scores over the records of one owner gives a much stronger signal "
                     "than a single record (dataset inference).</p>"))
    summary.update({"owners_with_2_known": int(len(sets)), "owners_pure_fraction": pure,
                    "test_with_known_owner": test_with_known_owner})

    # 4. Length.
    columns = 3
    rows = int(np.ceil(len(types) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, t in zip(axes.flat, types):
        part = known[known["source_type"] == t]
        lengths = part["length"].clip(lower=1)
        bins = np.geomspace(lengths.min(), lengths.max() + 1, 25)
        for label, color, name in ((1, MEMBER, "member"), (0, NONMEMBER, "non-member")):
            ax.hist(lengths[part["member"] == label], bins=bins, histtype="step", linewidth=2, color=color,
                    density=True, label=name)
        ax.set_xscale("log")
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(types):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.supxlabel(f"length of the content in {unit} (log scale)", color=INK2)
    fig.tight_layout()
    lengths = known.groupby(["source_type", "member"])["length"].median().unstack().reindex(types)
    lengths.columns = [f"median {unit}, non-member", f"median {unit}, member"]
    long = float((records.loc[records["kind"] == "text", "length"] > 2048).mean())
    sections.append(("4. Does the length tell the label?",
                     "<p>The content is <code>text</code> for texts (seen in S1) and <code>caption</code> for photos "
                     f"(seen in S3). {long:.1%} of the texts have more than 2048 {unit}. S1 cut them into more than one "
                     "training piece. If the two lines are different, a score that uses the length is a confounder.</p>"
                     + figure_html(fig, "length of members and non-members") + table_html(lengths.reset_index(), 0)))
    summary["median_length"] = {str(k): v for k, v in lengths.to_dict(orient="index").items()}
    summary["texts_above_2048"] = long

    # 5. Blind baselines.
    blind = baselines(known)
    blind.to_csv(out / "baselines.csv", index=False)
    best = blind[blind["records"] == "all"].sort_values("TPR@5%FPR", ascending=False).iloc[0]
    sections.append(("5. How good is a score without the model?",
                     "<p>These scores use only the length and the labels of the other known records of the same owner. "
                     "The owner prior is a fair score for test, because the owners of the sets have records in known and in test. "
                     f"The best score on all known records is <b>{html.escape(best['score'])}</b> with TPR@5%FPR "
                     f"{best['TPR@5%FPR']:.3f}. Random scores give approximately 0.05.</p>" + table_html(blind)))
    summary["baselines_all"] = blind[blind["records"] == "all"].set_index("score")[["TPR@5%FPR", "AUC"]].round(4).to_dict(orient="index")

    # 6. Text examples.
    body = ""
    for t in types:
        part = known[(known["source_type"] == t) & (known["kind"] == "text")]
        if part.empty:
            continue
        body += f"<h3>{html.escape(t)}</h3><div class='cards'>"
        for label in (1, 0):
            sample = part[part["member"] == label].head(1)
            for _, r in sample.iterrows():
                body += (f"<div class='card {'m' if label else 'n'}'><div class='badge'>{'member' if label else 'non-member'}"
                         f" · {html.escape(str(r['source']))} · {r['length']} {unit}</div><b>{short(r['title'], 120)}</b>"
                         f"<p>{short(r['content'], 500)}</p></div>")
        body += "</div>"
    sections.append(("6. What do the texts look like?", "<p>One member and one non-member of each source type, "
                     "from known. Only the start of each text is shown.</p>" + body))

    # 7. Photos.
    photos = records[records["kind"] == "photo"]
    body = ""
    if len(photos):
        sizes = photos["image_path"].dropna().map(lambda p: Image.open(p).size)
        hashes = photos["image_path"].dropna().map(lambda p: hashlib.md5(p.read_bytes()).hexdigest())
        size_table = sizes.map(lambda s: f"{s[0]} x {s[1]}").value_counts().head(8).rename_axis("size").reset_index(name="photos")
        summary.update({"photos": int(len(photos)), "photos_without_file": int(photos["image_path"].isna().sum()),
                        "photos_same_file": int(hashes.duplicated(keep=False).sum())})
        body += (f"<p>{len(photos)} photo records. {summary['photos_without_file']} records have no image file. "
                 f"{summary['photos_same_file']} records share an image file with a different record. "
                 "The model sees each photo as 49 image tokens at 224 x 224.</p>" + table_html(size_table))
        for t in [t for t in types if t in set(photos["source_type"])]:
            part = known[(known["source_type"] == t) & (known["kind"] == "photo") & known["image_path"].notna()]
            body += f"<h3>{html.escape(t)}</h3><div class='grid'>"
            for label in (1, 0):
                for _, r in part[part["member"] == label].head(6).iterrows():
                    body += (f"<figure class='{'m' if label else 'n'}'><img src='data:image/jpeg;base64,{thumbnail(r['image_path'])}'>"
                             f"<figcaption><span class='badge'>{'member' if label else 'non-member'}</span> "
                             f"{short(r['content'], 160)}</figcaption></figure>")
            body += "</div>"
    sections.append(("7. What do the photos look like?", body or "<p>No photo records.</p>"))

    # 8. Checks.
    content_dup = records["content"].duplicated(keep=False) & (records["content"] != "")
    texts = records[records["kind"] == "text"]
    title_start = texts.apply(lambda r: isinstance(r["title"], str) and r["content"].lstrip()[:200].startswith(r["title"].strip()[:60]), axis=1)
    title_rate = title_start.groupby(texts["source_type"]).mean().rename("title at the start of the text").reset_index()
    summary.update({"duplicate_content": int(content_dup.sum()), "title_at_start": float(title_start.mean())})
    sections.append(("8. Duplicates and titles",
                     f"<p>{int(content_dup.sum())} records have the same content as a different record. "
                     f"{title_start.mean():.0%} of the texts start with their title. The title of the other texts was not "
                     "a part of the training sequence.</p>" + table_html(title_rate)))

    # 9. sources.csv and the model files.
    model_files = sorted(p for p in root.rglob("*") if p.is_file() and "model" in p.relative_to(root).parts)
    files = pd.DataFrame({"file": [str(p.relative_to(root)) for p in model_files],
                          "MB": [p.stat().st_size / 1e6 for p in model_files]})
    sections.append(("9. sources.csv and the model files", table_html(sources) + table_html(files, 1)))
    script = next(root.rglob("openbumas.py"), None)
    if script:
        shutil.copy(script, out / "openbumas.py")

    findings = [
        f"{len(known)} known and {len(test)} test records. {known['member'].mean():.1%} of the known records are members.",
        f"For {pure:.0%} of the owners with two or more known records, all their known records have the same label.",
        f"{test_with_known_owner:.0%} of the test records have an owner with known records.",
        f"Best score without the model: {html.escape(best['score'])}, TPR@5%FPR {best['TPR@5%FPR']:.3f} on known.",
        f"{long:.1%} of the texts have more than 2048 {unit}.",
    ]
    page = PAGE.format(findings="".join(f"<li>{f}</li>" for f in findings),
                       body="".join(f"<section><h2>{title}</h2>{content}</section>" for title, content in sections))
    (out / "report.html").write_text(page, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return summary


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>P3 - data analysis</title><style>
body {{ font: 15px/1.5 system-ui, sans-serif; color: #191919; background: #fff; max-width: 1100px; margin: 0 auto; padding: 24px; color-scheme: light; }}
h1 {{ margin-bottom: 4px; }} h2 {{ border-top: 1px solid #e6e6e6; padding-top: 24px; margin-top: 40px; }}
.lead {{ background: #f4f7fc; border-left: 4px solid #257de0; padding: 12px 20px; }}
.fig {{ max-width: 100%; display: block; margin: 12px 0; }}
.tab {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
.tab th, .tab td {{ padding: 4px 10px; border-bottom: 1px solid #e6e6e6; text-align: left; }}
.tab th {{ color: #666; font-weight: 600; }}
.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.card {{ border: 1px solid #e6e6e6; border-radius: 6px; padding: 10px 14px; font-size: 13px; }}
.card.m, figure.m {{ border-top: 4px solid #257de0; }} .card.n, figure.n {{ border-top: 4px solid #f47230; }}
.badge {{ color: #666; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
figure {{ margin: 0; border: 1px solid #e6e6e6; border-radius: 6px; overflow: hidden; font-size: 12px; }}
figure img {{ width: 100%; display: block; }} figcaption {{ padding: 6px 8px; }}
</style></head><body>
<h1>P3: Did they train on my data? Data analysis</h1>
<p class="badge">Blue = member, orange = non-member, green = test. All labels come from known.jsonl.</p>
<div class="lead"><b>Key findings</b><ul>{findings}</ul></div>
{body}
<section><h2>10. Next steps</h2><ul>
<li>Run BUMAS-mini and Qwen3-0.6B-Base on all records and keep the log-probability of each token.</li>
<li>Compare LOSS, the reference ratio and Min-K%++ on known, separately for texts and photos.</li>
<li>Combine the scores of the records of one owner, and use the owner prior if section 5 shows a strong signal.</li>
</ul></section></body></html>"""
