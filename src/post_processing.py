"""
Aggregate predictions from multiple training splits and compute statistics.
"""
import os
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg


# ------------------------------------
# Util funcs
# ------------------------------------
def scores_boxplot(df: pd.DataFrame, x_name: str, y_name: str, hue_name: str,
                   figsize=(8, 5), outpath=None, title: str=None, ax=None):
    """
    Plot boxplots for various metrics that were calculated for multiple cv splits.
    Args:
        x_name : e.g. "metric"
        y_name : e.g. "smp", "Group"
        hue_name : e.g. "model"
    https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
    https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
    https://seaborn.pydata.org/generated/seaborn.boxplot.html
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Grouped boxplot and save it in a variable
    g = sns.boxplot(x=x_name, y=y_name, hue=hue_name, data=df,
                    palette="Set2", #["m", "g"],
                    linewidth=2);

    # Grouped stripplot and save it in a variable
    g = sns.stripplot(x=x_name, y=y_name, hue=hue_name, data=df,
                      jitter=True,
                      dodge=True,
                      # palette="Set2",
                      color="0.25",
                      marker="o",
                      alpha=0.25,
                      linewidth=1)

    # Set fontsize
    ticks_fontsize = 15
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=ticks_fontsize)
    ax.set_ylabel("Score", fontsize=ticks_fontsize)
    ax.set_xlabel(x_name, fontsize=ticks_fontsize)
    
    if title is not None:
        ax.set_title(title)
        
    # Legend
    handles, labels = g.get_legend_handles_labels()  # Legend info from the plot object
    n_labels = df[hue_name].nunique()
    l = plt.legend(handles[0:n_labels], labels[0:n_labels])  # Specify just one legend

    plt.tight_layout()
    ax.grid(True)

    if outpath is not None:
        plt.savefig(outpath, dpi=150)
        
    return ax


def scores_barplot(df: pd.DataFrame, x_name: str, y_name: str, hue_name: str,
                   col_name: str=None,
                   figsize=(8, 5), outpath=None, title: str=None, ax=None):
    """
    Plot barplots for various metrics that for specific cv splits.
    Args:
        x_name : e.g. "metric"
        y_name : e.g. "smp", "Group"
        hue_name : e.g. "model"
        col_name : e.g. "split"
    https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
    https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
    https://seaborn.pydata.org/generated/seaborn.boxplot.html
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    g = sns.barplot(x=x_name, y=y_name, hue=hue_name, data=df,
                    palette="Set2", ci="sd" # height=4, aspect=.7
                   );

    # Set fontsize
    ticks_fontsize = 15
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=ticks_fontsize)
    ax.set_ylabel("Score", fontsize=ticks_fontsize)
    ax.set_xlabel(x_name, fontsize=ticks_fontsize)
    
    if title is not None:
        ax.set_title(title)
        
    # Legend
    handles, labels = g.get_legend_handles_labels()  # Legend info from the plot object
    n_labels = df[hue_name].nunique()
    l = plt.legend(handles[0:n_labels], labels[0:n_labels])  # Specify just one legend

    plt.tight_layout()
    ax.grid(True)

    if outpath is not None:
        plt.savefig(outpath, dpi=150)
        
    return ax


# def scores_barplot(df: pd.DataFrame, x_name: str, y_name: str, hue_name: str,
#                    col_name: str=None,
#                    figsize=(8, 5), outpath=None, title: str=None, ax=None):
#     """
#     Plot barplots for various metrics that for specific cv splits.
#     Args:
#         x_name : e.g. "metric"
#         y_name : e.g. "smp", "Group"
#         hue_name : e.g. "model"
#         col_name : e.g. "split"
#     https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
#     https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/
#     https://seaborn.pydata.org/generated/seaborn.boxplot.html
#     """
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#     g = sns.barplot(x=x_name, y=y_name, hue=hue_name, data=df,
#                     palette="Set2" # height=4, aspect=.7
#                    );

#     # Set fontsize
#     ticks_fontsize = 15
#     g.set_xticklabels(g.get_xmajorticklabels(), fontsize=ticks_fontsize)
#     ax.set_ylabel("Score", fontsize=ticks_fontsize)
#     ax.set_xlabel(x_name, fontsize=ticks_fontsize)
    
#     if title is not None:
#         ax.set_title(title)
        
#     # Legend
#     handles, labels = g.get_legend_handles_labels()  # Legend info from the plot object
#     n_labels = df[hue_name].nunique()
#     l = plt.legend(handles[0:n_labels], labels[0:n_labels])  # Specify just one legend

#     plt.tight_layout()
#     ax.grid(True)

#     if outpath is not None:
#         plt.savefig(outpath, dpi=150)
        
#     return ax


def t_test(scores, splits=None, met="PRC-AUC", agg_by="smp"):
    if splits is not None:
        scores = scores[scores["split"].isin(splits)]
        
    a = scores[(scores["metric"] == met) & (scores["model"] == "NN: Tile-GE-DD")][agg_by]
    b = scores[(scores["metric"] == met) & (scores["model"] == "NN: GE-DD")][agg_by]
    c = scores[(scores["metric"] == met) & (scores["model"] == "LGBM: GE-DD")][agg_by]
    
    print(stats.shapiro(a))
    print(stats.shapiro(b))
    print(stats.shapiro(c))
    
    print(stats.ttest_rel(a, b, axis=0, alternative="two-sided"))
    print(stats.ttest_rel(a, c, axis=0, alternative="two-sided"))
    print(stats.ttest_rel(b, c, axis=0, alternative="two-sided"))


def main(args):
    # ------------------------------------
    # Multimodal
    # ------------------------------------
    dataname = "tidy_drug_pairs_all_samples"
    prjname = "bin_rsp_drug_pairs_all_samples"
    fname = "test_scores.csv"

    datadir = fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd"
    prfx = "split_"
    splits_dir_list = sorted(datadir.glob(f"{prfx}*"))

    all_te_scores = []
    missing_scores = []
    for split_dir in splits_dir_list:
        split_id = str(split_dir.name).split(prfx)[1].split("_")[0]
        fpath = split_dir/fname
        if (fpath).exists():
            te_scores = pd.read_csv(fpath)
            te_scores["split"] = int(split_id)
            all_te_scores.append(te_scores)
        else:
            missing_scores.append(split_id)

    print(f"Test scores were found for these splits: {missing_scores}")
    mm = pd.concat(all_te_scores, axis=0).sort_values("split").reset_index(drop=True)
    mm = mm.rename(columns={"pred_for": "metric"})
    del fpath, te_scores, all_te_scores

    # ------------------------------------
    # Single-modal
    # ------------------------------------
    dataname = "tidy_drug_pairs_all_samples"
    prjname = "bin_rsp_drug_pairs_all_samples"
    fname = "test_keras_scores.csv"

    datadir = fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd"
    prfx = "split_"
    splits_dir_list = sorted(datadir.glob(f"{prfx}*"))

    all_te_scores = []
    missing_scores = []
    for split_dir in splits_dir_list:
        split_id = str(split_dir.name).split(prfx)[1].split("_")[0]
        fpath = split_dir/fname
        if fpath.exists():
            te_scores = pd.read_csv(fpath)
            te_scores["split"] = int(split_id)
            all_te_scores.append(te_scores)
        else:
            missing_scores.append(split_id)

    print(f"Test scores were found for these splits: {missing_scores}")
    sm = pd.concat(all_te_scores, axis=0).sort_values("split").reset_index(drop=True)
    sm = sm.rename(columns={"pred_for": "metric"})
    del fpath, te_scores, all_te_scores

    # ------------------------------------
    # LGBM
    # ------------------------------------
    lgb_datadir = fdir/"../data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31"
    fname = "te_scores.csv"
    prfx = "cv_"
    splits_dir_list = sorted(lgb_datadir.glob(f"{prfx}*"))

    all_te_scores = []
    missing_scores = []
    for split_dir in splits_dir_list:
        split_id = str(split_dir.name).split(prfx)[1].split("_")[0]
        fpath = split_dir/fname
        if (fpath).exists():
            te_scores = pd.read_csv(fpath)
            te_scores["split"] = int(split_id)
            all_te_scores.append(te_scores)
        else:
            missing_scores.append(split_id)

    print(f"Test scores were found for these splits: {missing_scores}")
    lgb = pd.concat(all_te_scores, axis=0).sort_values("split").reset_index(drop=True)
    del fpath, te_scores, all_te_scores

    # ------------------------------------
    # Agg scores from all models
    # ------------------------------------
    # import ipdb; ipdb.set_trace()
    # Keep common metrics
    s1 = set(mm["metric"].values)
    s2 = set(sm["metric"].values)
    s3 = set(lgb["metric"].values)
    common_metrics = list(reduce(set.intersection, [s1, s2, s3]))
    
    mm = mm[mm["metric"].isin(common_metrics)].sort_values(["split", "metric"])
    sm = sm[sm["metric"].isin(common_metrics)].sort_values(["split", "metric"])
    lgb = lgb[lgb["metric"].isin(common_metrics)].sort_values(["split", "metric"])
    
    mm["model"] = "NN: Tile-GE-DD"
    sm["model"] = "NN: GE-DD"
    lgb["model"] = "LGBM: GE-DD"

    # Rename items
    def rename_items_of_single_col(x, mapper_dict):
        """ mapper_dict where keys and values are the old and new values, respectively. """
        for k, v in mapper_dict.items():
            if k == x:
                return mapper_dict[k]
        return x

    mapper = {"brier": "Brier", "mcc": "MCC", "pr_auc": "PRC-AUC", "roc_auc": "ROC-AUC"}
    mm["metric"] = mm["metric"].map(lambda x: rename_items_of_single_col(x, mapper))
    sm["metric"] = sm["metric"].map(lambda x: rename_items_of_single_col(x, mapper))
    lgb["metric"] = lgb["metric"].map(lambda x: rename_items_of_single_col(x, mapper))

    # Agg scores for each metric across the splits
    def agg_metrics(df):
        df = df.groupby("metric").agg(smp_mean=("smp", "mean"), # smp_std=("smp", "std"),
                                      grp_mean=("Group", "mean"), # grp_std=("Group", "std")
                                      ).reset_index()
        return df

    mm_res = agg_metrics(mm)
    sm_res = agg_metrics(sm)
    lgb_res = agg_metrics(lgb)

    # Print
    print("\nNN: Tile-GE-DD")
    print(mm_res)
    print("\nNN: GE-DD")
    print(sm_res)
    print("\nLGBM: GE-DD")
    print(lgb_res)

    # Plot
    cols = ["metric", "smp", "Group", "split", "model"]
    all_scores = pd.concat([mm[cols], sm[cols], lgb[cols]], axis=0).reset_index(drop=True)
    
    outdir = fdir/"../projects/bin_rsp_drug_pairs_all_samples"
    ax = scores_boxplot(df=all_scores, x_name="metric", y_name="smp", hue_name="model",
                        title="Per-sample scores", outpath=outdir/"boxplot_smp_scores.png")
    ax = scores_boxplot(df=all_scores, x_name="metric", y_name="Group", hue_name="model",
                        title="Per-group scores", outpath=outdir/"boxplot_grp_scores.png")
   
    ax = scores_barplot(df=all_scores, x_name="metric", y_name="smp", hue_name="model",
                        title=f"Per-sample scores", outpath=outdir/"barplot_smp_scores.png")
    ax = scores_barplot(df=all_scores, x_name="metric", y_name="Group", hue_name="model",
                        title=f"Per-group scores", outpath=outdir/"barplot_grp_scores.png" )

#     split = 81
#     agg_by = "smp"
#     ax = scores_barplot(df=all_scores[all_scores["split"].isin([split])],
#                         x_name="metric", y_name=agg_by, hue_name="model",
#                         title=f"Per-sample scores (split {split})",
#                         outpath=outdir/f"barplot_smp_scores_split_{split}.png")
#     agg_by = "Group"
#     ax = scores_barplot(df=all_scores[all_scores["split"].isin([split])],
#                         x_name="metric", y_name=agg_by, hue_name="model",
#                         title=f"Per-group scores (split {split})",
#                         outpath=outdir/f"barplot_grp_scores_split_{split}.png")
#     split = 99
#     agg_by = "smp"
#     ax = scores_barplot(df=all_scores[all_scores["split"].isin([split])],
#                         x_name="metric", y_name=agg_by, hue_name="model",
#                         title=f"Per-sample scores (split {split})",
#                         outpath=outdir/f"barplot_smp_scores_split_{split}.png")
#     agg_by = "Group"
#     ax = scores_barplot(df=all_scores[all_scores["split"].isin([split])],
#                         x_name="metric", y_name=agg_by, hue_name="model",
#                         title=f"Per-group scores (split {split})",
#                         outpath=outdir/f"barplot_grp_scores_split_{split}.png")
    
    # Stat significance
    print("\nStatistical significance:")
    t_test(all_scores, met="PRC-AUC", agg_by="smp")

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])