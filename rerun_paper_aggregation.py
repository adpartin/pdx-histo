"""
Re-execute the aggregation of nbs/post-processing.ipynb cells 41/46/48 with
the Oct 2021 MM-Net sweep path (instead of the May 2021 path that's saved in
the notebook). Compares per-model mean smp-level metrics across 100 splits
against the published table.

Run with the paper-era env:
    /homes/apartin/miniconda3/envs/pdx_lamina/bin/python rerun_paper_aggregation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src.post_processing import agg_scores_from_splits  # noqa: E402

PROJ = REPO / "projects" / "bin_rsp_drug_pairs_all_samples"
LGBM = REPO / "data" / "PDX_Transfer_Learning_Classification" / "Results_MultiModal_Learning" / "1.0_True_False_100_31"


def load(datadir, fname, prfx, model_name, rename_pred_for=False):
    df = agg_scores_from_splits(datadir=datadir, fname=fname, prfx=prfx, print_fn=lambda *_: None)
    if rename_pred_for:
        df = df.rename(columns={"pred_for": "metric"})
    df["model"] = model_name
    return df


# ---- Cell 41 (with Oct path for MM-Net) ----
mm  = load(PROJ,                      "test_scores.csv",         "split_", "mm-NN",    rename_pred_for=True)
umh = load(PROJ / "runs_tile_dd",     "test_scores.csv",         "split_", "umh-NN",   rename_pred_for=True)
ume = load(PROJ / "runs_ge_dd",       "test_keras_scores.csv",   "split_", "ume-NN")
lgb = load(LGBM,                      "te_scores.csv",           "cv_",    "ume-LGBM")

# Older MM-Net (May 2021) for side-by-side comparison
mm_may = load(PROJ / "runs_tile_ge_dd", "test_scores.csv",        "split_", "mm-NN_may", rename_pred_for=True)

all_scores = pd.concat([mm, umh, ume, lgb, mm_may], axis=0, ignore_index=True, sort=False)

# Drop duplicate ap_macro rows that the Oct test_scores.csv contains
all_scores = all_scores.drop_duplicates(subset=["split", "model", "metric"], keep="first").reset_index(drop=True)


# ---- Per-model mean (smp-level) per metric ----
metric_map = {"mcc": "MCC", "pr_auc": "AUPRC", "roc_auc": "AUROC"}
PAPER = {
    "mm-NN":    {"MCC": 0.3102, "AUPRC": 0.2974, "AUROC": 0.7978},
    "ume-NN":   {"MCC": 0.2958, "AUPRC": 0.2996, "AUROC": 0.8047},
    "umh-NN":   {"MCC": 0.2124, "AUPRC": 0.2303, "AUROC": 0.7977},
    "ume-LGBM": {"MCC": 0.2594, "AUPRC": 0.2784, "AUROC": 0.8065},
}

def summary_table(agg_col, agg_fn, agg_label):
    print()
    print("=" * 96)
    print(f"Per-model {agg_label} across 100 splits ({agg_col}-level)  —  paper value in parens")
    print("=" * 96)
    hdr = f"{'model':<14}{'n':>5}    " + "  ".join(f"{m:>18}" for m in metric_map.values())
    print(hdr)
    for model in ["mm-NN", "mm-NN_may", "ume-NN", "umh-NN", "ume-LGBM"]:
        sub = all_scores[all_scores["model"] == model]
        n = sub["split"].nunique()
        cells = [f"{model:<14}{n:>5}   "]
        for met_csv, met_label in metric_map.items():
            vals = sub[sub["metric"] == met_csv][agg_col].astype(float)
            observed = agg_fn(vals)
            paper_key = "mm-NN" if model == "mm-NN_may" else model
            paper_val = PAPER.get(paper_key, {}).get(met_label, None)
            cells.append(f"  {observed:.4f} ({paper_val:.4f})" if paper_val is not None else f"  {observed:.4f}         ")
        print("".join(cells))

summary_table("smp",   np.mean,   "MEAN")
summary_table("smp",   np.median, "MEDIAN")
summary_table("Group", np.mean,   "MEAN")
summary_table("Group", np.median, "MEDIAN")


# ---- Cells 46/48: paired t-tests, mm-NN vs each baseline ----
print()
print("=" * 78)
print("Paired t-test (mm-NN vs baselines, scipy.stats.ttest_rel two-sided, smp-level)")
print("=" * 78)

def pair_vec(model, met, agg_col="Group"):
    sub = all_scores[(all_scores["model"] == model) & (all_scores["metric"] == met)]
    sub = sub.sort_values("split")
    return sub[agg_col].astype(float).values, sub["split"].values

def ttest_table(reference_model, label):
    print()
    print("=" * 96)
    print(f"Paired t-test ({label} vs baselines, scipy.stats.ttest_rel two-sided, Group-level)")
    print("=" * 96)
    print(f"{'comparison':<32}{'metric':>10}    {'mean_diff':>12}{'t_stat':>10}{'p_value':>12}{'n_splits':>10}")
    for baseline in [m for m in ["mm-NN", "mm-NN_may", "ume-NN", "umh-NN", "ume-LGBM"] if m != reference_model]:
        for met_csv, met_label in metric_map.items():
            mm_vec, mm_splits = pair_vec(reference_model, met_csv)
            bl_vec, bl_splits = pair_vec(baseline, met_csv)
            common = sorted(set(mm_splits) & set(bl_splits))
            mm_aligned = pd.Series(mm_vec, index=mm_splits).loc[common].values
            bl_aligned = pd.Series(bl_vec, index=bl_splits).loc[common].values
            t, p = stats.ttest_rel(mm_aligned, bl_aligned, axis=0, alternative="two-sided")
            diff = (mm_aligned - bl_aligned).mean()
            sig = " *" if p < 0.05 else ""
            print(f"{reference_model + ' vs ' + baseline:<32}{met_label:>10}    {diff:>+12.4f}{t:>10.3f}{p:>12.4f}{len(common):>10}{sig}")

ttest_table("mm-NN_may", "mm-NN_may = the May 2021 sweep (paper's likely source)")
ttest_table("mm-NN",     "mm-NN = the Oct 2021 sweep (saved-state in notebook but worse)")

# Wins per metric (paper claim: MM-Net beat ume-NN on 46/100 splits)
print()
print("=" * 96)
print("Splits where mm-NN_may (paper MM-Net) beats each baseline (Group-level)")
print("=" * 96)
print("(Paper claims: 'In 46/100 splits, MM-Net exceeded UME-Net baseline performance')")
for baseline in ["ume-NN", "umh-NN", "ume-LGBM"]:
    for met_csv, met_label in metric_map.items():
        mm_vec,  mm_splits  = pair_vec("mm-NN_may", met_csv, agg_col="Group")
        bl_vec,  bl_splits  = pair_vec(baseline,    met_csv, agg_col="Group")
        common = sorted(set(mm_splits) & set(bl_splits))
        mm_aligned = pd.Series(mm_vec, index=mm_splits).loc[common].values
        bl_aligned = pd.Series(bl_vec, index=bl_splits).loc[common].values
        wins = int((mm_aligned > bl_aligned).sum())
        print(f"  mm-NN_may > {baseline:<10} on {met_label:<6}  : {wins}/{len(common)}")
