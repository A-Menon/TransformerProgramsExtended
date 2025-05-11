#!/usr/bin/env python3
import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────
BASE_DIR = "output/rasp"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# preserve the paper’s task order
TASKS = ["double_hist", "hist", "reverse", "sort", "most_freq"]
# mapping for module names → pretty labels
MODULE_LABEL = {
    "prefix":  "Prefix Sum",
    "expert":  "Mixture of Experts",
    "both":    "Prefix and Experts",
    "improved":"Improved",
}
# mapping for variant → true hyperparam settings
VARIANT_LABEL = {
    "baseline": "|V|=8, N=8",
    "longlen":  "|V|=8, N=16",
    "bigvocab": "|V|=16, N=16",
}

# ─── 1) Gather final‐epoch results ─────────────────────────────────────────
pattern = os.path.join(
    BASE_DIR, "*", "*", "modules_*",
    "k*_len*_L*_H*_M*", "s*", "results.csv"
)
files = glob.glob(pattern, recursive=True)
meta_re = re.compile(
    r".*/rasp/(?P<dataset>[^/]+)/"
    r"(?P<variant>[^/]+)/modules_(?P<module>[^/]+)/"
    r"k(?P<k>\d+)_len(?P<maxlen>\d+)_L(?P<L>\d+)_H(?P<H>\d+)_M(?P<M>\d+)/"
    r"s(?P<seed>\d+)/results\.csv$"
)

records = []
for fp in files:
    m = meta_re.match(fp.replace("\\","/"))
    if not m:
        continue
    md = m.groupdict()
    df = pd.read_csv(fp)
    final_ep = df["epoch"].max()
    df = df[
        (df["epoch"]     == final_ep) &
        (df["split"]     == "test")     &
        (df["sample_fn"] == "argmax")
    ][["acc"]]
    df = df.assign(
        dataset = md["dataset"],
        variant = md["variant"],
        module  = md["module"],
        acc     = df["acc"]*100  # to percent
    )
    records.append(df)

all_df = pd.concat(records, ignore_index=True)

# ─── 2) Pivot wide and compute Improved ────────────────────────────────────
wide = all_df.pivot_table(
    index=["dataset","variant"],
    columns="module",
    values="acc"
).reset_index()

# ensure every column exists
for col in ("none","prefix","expert","both"):
    if col not in wide:
        wide[col] = 0.0
wide["improved"] = wide[["prefix","expert","both"]].max(axis=1)

# add pretty labels
wide["VariantLabel"] = wide["variant"].map(VARIANT_LABEL)
wide["Dataset"]      = wide["dataset"].astype("category")
wide["Dataset"] = wide["Dataset"].cat.set_categories(TASKS, ordered=True)

# ─── 3) Bar charts ─────────────────────────────────────────────────────────
for module_key in ("prefix","expert","both","improved"):
    plt.figure(figsize=(20,4))
    for i, ds in enumerate(TASKS):
        ax = plt.subplot(1,5,i+1)
        sub = wide[wide["dataset"]==ds]
        # melt to long form: Base vs Selected module
        dfm = pd.DataFrame({
            "Group":    ["Base"]*len(sub) + [MODULE_LABEL[module_key]]*len(sub),
            "Variant":  list(sub["VariantLabel"])*2,
            "Accuracy": list(sub["none"]) + list(sub[module_key])
        })
        sns.barplot(
            data=dfm,
            x="Group",
            y="Accuracy",
            hue="Variant",
            hue_order=["|V|=8, N=8", "|V|=8, N=16", "|V|=16, N=16"],
            ax=ax
        )
        ax.set_title(ds.replace("_"," ").title())
        if i==0:
            ax.set_ylabel("Test Accuracy (%)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.get_legend().remove()
    plt.suptitle(f"Base vs {MODULE_LABEL[module_key]} Test Accuracy", y=0.98)
    # single legend on right
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, title="Hyperparam", loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fname = f"bar_base_vs_{module_key}.png"
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()

# ─── 4) Scatter: Base vs Improved by Task ─────────────────────────────────
scatter_df = wide.copy()
jitter_scale = 0.3  # adjust to taste
scatter_df["base_j"]     = scatter_df["none"]     + np.random.uniform(-jitter_scale, jitter_scale, size=len(scatter_df))
scatter_df["improved_j"] = scatter_df["improved"] + np.random.uniform(-jitter_scale, jitter_scale, size=len(scatter_df))
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=scatter_df,
    x="base_j", y="improved_j",
    hue="dataset",
    s=100,
    edgecolor="w",      # white border
    linewidth=0.5,
    alpha=0.75,         # 25% transparent
    palette="tab10"
)

x = scatter_df["base_j"]
y = scatter_df["improved_j"]
pad = max(x.max() - x.min(), y.max() - y.min()) * 0.05
lims = [min(x.min(), y.min()) - pad, max(x.max(), y.max()) + pad]
plt.plot(lims, lims, "--", color="gray", linewidth=1)

plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Base Accuracy (%)")
plt.ylabel("Improved Accuracy (%)")
plt.title("Base vs Best‐of‐All Improvements")
plt.legend(title="Task", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "scatter_base_vs_improved_jitter.png"), dpi=300)
plt.close()

# ─── 5) Graphic summary table ─────────────────────────────────────────────
# build a flat table
table_df = wide[[
    "dataset","VariantLabel","none","prefix","expert","both","improved"
]].copy()
table_df.columns = [
    "Task","Hyperparam","Base","Prefix Sum",
    "Mixture of Experts","Prefix and Experts","Improved"
]
# sort
table_df["Task"] = pd.Categorical(table_df["Task"], categories=TASKS, ordered=True)
table_df["Hyperparam"] = pd.Categorical(
    table_df["Hyperparam"],
    categories=[VARIANT_LABEL[v] for v in ("baseline","longlen","bigvocab")],
    ordered=True
)
table_df = table_df.sort_values(["Task","Hyperparam"])

for col in ["Base", "Prefix Sum", "Mixture of Experts", "Prefix and Experts", "Improved"]:
    table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}")

# render with Matplotlib table
fig, ax = plt.subplots(
    figsize=(10, 0.5 * len(table_df) + 1)
)
ax.axis("off")
tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="upper left"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "summary_table.png"), dpi=300)
plt.close()

print("All figures and table saved in", FIG_DIR)