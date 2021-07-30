"""
Aggregate predictions from multiple training splits and compute statistics.
"""
import os
import sys
from functools import reduce
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src.utils.classlogger import Logger
from src.utils.utils import get_print_func


# ------------------------------------
# Util funcs
# ------------------------------------
def agg_scores_from_splits(datadir, fname: str, prfx: str, print_fn=print):
    """
    Args:
        datadir : path that contains the split dirs (e.g., runs_tile_ge_dd)
        fname : file name that contains the scores (e.g., test_scores.csv)
        prfx : prefix that indicates the split dir (e.g., split_0_tile_ge_...)
    Returns:
        df : table of aggregated scores
    """
    splits_dir_list = sorted(datadir.glob(f"{prfx}*"))

    all_te_scores = []
    missing_scores = []
    
    # Iterate over split dirs
    for split_dir in splits_dir_list:
        split_id = str(split_dir.name).split(prfx)[1].split("_")[0]
        fpath = split_dir/fname
        if fpath.exists():
            te_scores = pd.read_csv(fpath)
            te_scores["split"] = int(split_id)
            all_te_scores.append(te_scores)
        else:
            missing_scores.append(split_id)

    print_fn(f"Test scores were not found for these splits: {missing_scores}")
    df = pd.concat(all_te_scores, axis=0).sort_values("split").reset_index(drop=True)
    return df


def get_agg_scores(scores, agg_method: str="mean"):
    """ Aggregate scores for each metric across the splits for smp and grp scores. """
    # Agg scores for each metric across the splits
    agg_scores = scores.groupby(["model", "metric"]).agg(
        smp_mean=("smp", agg_method),
        grp_mean=("Group", agg_method)
    ).reset_index()
    smp_scores = pd.pivot_table(agg_scores, values="smp_mean", index="metric", columns="model").reset_index()
    smp_scores.columns.name = None
    grp_scores = pd.pivot_table(agg_scores, values="grp_mean", index="metric", columns="model").reset_index()
    grp_scores.columns.name = None
    return smp_scores, grp_scores


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
    # ax.set_xlabel(x_name, fontsize=ticks_fontsize)
    ax.set_xlabel(None)
    
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
    # ax.set_xlabel(x_name, fontsize=ticks_fontsize)
    ax.set_xlabel(None)
    
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


def make_plots(df, outdir, plot_name):
    # Boxplot
    ax = scores_boxplot(df, x_name="metric", y_name="smp", hue_name="model",
                        title="Per-sample scores", outpath=outdir/f"boxplot_smp_scores_{plot_name}.png")
    ax = scores_boxplot(df, x_name="metric", y_name="Group", hue_name="model",
                        title="Per-group scores", outpath=outdir/f"boxplot_grp_scores_{plot_name}.png")
    # Barplot
    ax = scores_barplot(df, x_name="metric", y_name="smp", hue_name="model",
                        title=f"Per-sample scores", outpath=outdir/f"barplot_smp_scores_{plot_name}.png")
    ax = scores_barplot(df, x_name="metric", y_name="Group", hue_name="model",
                        title=f"Per-group scores", outpath=outdir/f"barplot_grp_scores_{plot_name}.png")
    return None


def t_test(scores, splits: Optional[List[int]]=None, met: str="PRC-AUC", agg_by: str="smp"):
    """
    Args:
        splits : list of split ids to consider when performing t-test
    Returns:
        dd : dict with p-values calculated using paired t-test for all model combinations
    """
    if splits is not None:
        scores = scores[scores["split"].isin(splits)]
        
    # Must sort by split!
    scores = scores.sort_values(["metric", "split"], ascending=True).reset_index(drop=True)

    # mm_vec = scores[(scores["metric"] == met) & (scores["model"] == "mm-NN")][agg_by]
    # um_vec = scores[(scores["metric"] == met) & (scores["model"] == "ume-NN")][agg_by]
    # lgbm_vec = scores[(scores["metric"] == met) & (scores["model"] == "ume-LGBM")][agg_by]
    
    # print_fn(stats.shapiro(mm_vec))
    # print_fn(stats.shapiro(um_vec))
    # print_fn(stats.shapiro(lgbm_vec))

    # print_fn("mmNN-umNN  {}".format(stats.ttest_rel(mm_vec, um_vec, axis=0, alternative="two-sided")))
    # print_fn("mmNN-LGBM {}".format(stats.ttest_rel(mm_vec, lgbm_vec, axis=0, alternative="two-sided")))
    # print_fn("umNN-LGBM {}".format(stats.ttest_rel(um_vec, lgbm_vec, axis=0, alternative="two-sided")))

    # dd0 = {}
    # _, p_value = stats.ttest_rel(mm_vec, um_vec, axis=0, alternative="two-sided");   dd0["p-value: mm-NN vs ume-NN"] = p_value
    # _, p_value = stats.ttest_rel(mm_vec, lgbm_vec, axis=0, alternative="two-sided"); dd0["p-value: mm-NN vs ume-LGBM"] = p_value
    # _, p_value = stats.ttest_rel(um_vec, lgbm_vec, axis=0, alternative="two-sided"); dd0["p-value: ume-NN vs ume-LGBM"] = p_value

    m_vecs = {}
    models = []
    for model in scores["model"].unique():
        models.append(model)
        m_vecs[model] = scores[(scores["metric"] == met) & (scores["model"] == model)][agg_by].values

    import itertools
    models_combs = itertools.combinations(models, r=2)

    # import ipdb; ipdb.set_trace()
    dd = {}
    for model_pair in models_combs:
        v1 = m_vecs[model_pair[0]]
        v2 = m_vecs[model_pair[1]]
        _, p_value = stats.ttest_rel(v1, v2, axis=0, alternative="two-sided");
        dd[f"p-value: {model_pair[0]} vs {model_pair[1]}"] = p_value

    return dd


def t_test_all_metrics(scores, splits: Optional[List[int]]=None, agg_by: str="smp"):
    """
    Args:
        splits : list of split ids to consider when performing t-test
    """
    if splits is not None:
        scores = scores[scores["split"].isin(splits)]
        
    pmat = {}
    for met in scores["metric"].unique():
        met_pv = t_test(scores, met=met, agg_by=agg_by)
        pmat[met] = met_pv

    pmat = pd.DataFrame(pmat)
    pmat = pmat.T.reset_index().rename(columns={"index": "metric"})
    pmat = pmat.round(5)
    return pmat


def main(args):

    # import ipdb; ipdb.set_trace()
    outdir = fdir/"../projects/bin_rsp_drug_pairs_all_samples"

    # Logger
    lg = Logger(outdir/"logger.log")
    print_fn = get_print_func(lg.logger)
    print_fn(f"File path: {fdir}")

    # ------------------------------------
    # Multimodal - mm-NN
    # ------------------------------------
    mm = agg_scores_from_splits(
        datadir=fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd",
        fname="test_scores.csv", prfx="split_", print_fn=print_fn)
    mm = mm.rename(columns={"pred_for": "metric"})
    mm["model"] = "mm-NN"  # Create "model" column
    
    # ------------------------------------
    # Unimodal images - umh-NN
    # ------------------------------------
    umh = agg_scores_from_splits(
        datadir=fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_tile_dd",
        fname="test_scores.csv", prfx="split_", print_fn=print_fn)
    umh["model"] = "umh-NN"  # Create "model" column

    # ------------------------------------
    # Unimodal gene expression - ume-NN
    # ------------------------------------
    ume = agg_scores_from_splits(
        datadir=fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd",
        fname="test_keras_scores.csv", prfx="split_", print_fn=print_fn)
    ume["model"] = "ume-NN"  # Create "model" column

    # ------------------------------------
    # Unimodal gene expression - ume-NN drop aug
    # ------------------------------------
    ume_drop_aug = agg_scores_from_splits(
        datadir=fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd_drop_aug",
        fname="test_keras_scores.csv", prfx="split_", print_fn=print_fn)
    ume_drop_aug["model"] = "ume-NN-drop-aug"  # Create "model" column
    
    # ------------------------------------
    # Unimodal gene expression - ume-NN only pairs
    # ------------------------------------
    ume_only_pairs = agg_scores_from_splits(
        datadir=fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd_only_pairs",
        fname="test_keras_scores.csv", prfx="split_", print_fn=print_fn)
    ume_only_pairs["model"] = "ume-NN-only-pairs"  # Create "model" column
    
    # ------------------------------------
    # LGBM
    # ------------------------------------
    lgbm = agg_scores_from_splits(
        datadir=fdir/"../data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31",
        fname="te_scores.csv", prfx="cv_", print_fn=print_fn)
    lgbm["model"] = "ume-LGBM"  # Create "model" column

    # ------------------------------------
    # Agg scores from all models
    # ------------------------------------
    # Concat scores from all models
    cols = ["metric", "smp", "Group", "split", "model"]
    all_scores = pd.concat([mm[cols], ume[cols], lgbm[cols], ume_drop_aug[cols], ume_only_pairs[cols]],
                           axis=0).reset_index(drop=True)

    # Keep common metrics
    # s1 = set(mm["metric"].values)
    # s2 = set(ume["metric"].values)
    # s3 = set(lgbm["metric"].values)
    # s4 = set(ume_drop_aug["metric"].values)
    # s5 = set(ume_only_pairs["metric"].values)
    # common_metrics = list(reduce(set.intersection, [s1, s2, s3, s4, s5]))
    df_list = [mm, umh, ume, lgbm, ume_drop_aug, ume_only_pairs]
    ll = [df["metric"].values for df in df_list]
    common_metrics = list(reduce(set.intersection, ll))
    all_scores = all_scores[all_scores["metric"].isin(common_metrics)].reset_index(drop=True)

    # Rename metric names
    def rename_items_of_single_col(x, mapper_dict):
        """ mapper_dict where keys and values are the old and new values, respectively. """
        for k, v in mapper_dict.items():
            if k == x:
                return mapper_dict[k]
        return x

    mapper = {"brier": "Brier", "mcc": "MCC", "pr_auc": "PRC-AUC", "roc_auc": "ROC-AUC"}
    all_scores["metric"] = all_scores["metric"].map(lambda x: rename_items_of_single_col(x, mapper))

    # Drop metrics
    met_drop = ["Brier"]
    all_scores = all_scores[~all_scores["metric"].isin(met_drop)]
    all_scores = all_scores.sort_values(["metric", "split"], ascending=True)

    # Save
    all_scores.to_csv(outdir/"all_scores.csv", index=False)


    # Analysis for mm-comp
    name = "mm-comp"
    model_names = ["mm-NN", "umh-NN", "ume-NN", "ume-LGBM"]
    scores = all_scores[all_scores["model"].isin(model_names)].reset_index(drop=True)
    make_plots(scores, outdir, plot_name=name)
    smp_scores, grp_scores = get_agg_scores(scores, agg_method="mean")

    # Perform paired t-test between scores (of different splits) of different models
    # and create table that summarises the mean of scores across splits and p-values
    smp_pmat = t_test_all_metrics(scores, agg_by="smp")
    smp_res = smp_scores.merge(smp_pmat, on="metric", how="inner")

    grp_pmat = t_test_all_metrics(scores, agg_by="Group")
    grp_res = grp_scores.merge(grp_pmat, on="metric", how="inner")

    smp_res.to_csv(outdir/f"smp_res_{name}.csv", index=False)
    grp_res.to_csv(outdir/f"grp_res_{name}.csv", index=False)


    # Analysis for ume-NN
    name = "ume-NN"
    model_names = ["ume-NN", "ume-NN-drop-aug", "ume-NN-only-pairs"]
    scores = all_scores[all_scores["model"].isin(model_names)].reset_index(drop=True)
    make_plots(scores, outdir, plot_name=name)
    smp_scores, grp_scores = get_agg_scores(scores, agg_method="mean")

    # Perform paired t-test between scores (of different splits) of different models
    # and create table that summarises the mean of scores across splits and p-values
    smp_pmat = t_test_all_metrics(scores, agg_by="smp")
    smp_res = smp_scores.merge(smp_pmat, on="metric", how="inner")

    grp_pmat = t_test_all_metrics(scores, agg_by="Group")
    grp_res = grp_scores.merge(grp_pmat, on="metric", how="inner")

    smp_res.to_csv(outdir/f"smp_res_{name}.csv", index=False)
    grp_res.to_csv(outdir/f"grp_res_{name}.csv", index=False)


    print_fn("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
