"""
Exploratory analysis.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg


# ------------------------------------
# Util funcs
# ------------------------------------
def barplot_sample_count(resp_dict: dict, title: str=None, ylabel: str=None,
                         ticks_fontsize: int=15, figsize=(10, 8)):
    """ Plots the count (barplot) of responses for each unique ctype (or other variable).
    stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
    matplotlib.org/2.0.2/examples/api/barchart_demo.html
    """
    labels = resp_dict["labels"]
    pos = np.arange(len(labels))  # the center locations for the bars
    bar_width = 0.4    # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.barh(y=pos, width=resp_dict[0], height=bar_width, color="g", label="Non-response")
    rects2 = ax.barh(y=pos + bar_width, width=resp_dict[1], height=bar_width, color="b", label="Response")

    # ticks_fontsize = 15
    if title:
        ax.set_title(title)    
    if ylabel:
        # ax.set_ylabel("PDX Cancer Type", fontsize=ticks_fontsize)
        ax.set_ylabel(ylabel, fontsize=ticks_fontsize)
    ax.set_xlabel('Count', fontsize=ticks_fontsize)
    ax.set_yticks(pos + bar_width / 2)
    ax.set_yticklabels(labels, fontsize=ticks_fontsize);
    ax.tick_params(axis="x", labelsize=ticks_fontsize)

    align = "left"
    xloc = 5
    
    for rect in rects1:
        width = rect.get_width()
        width = np.around(width, decimals=2)

        yloc = rect.get_y() + rect.get_height() / 2  # Center the text vertically in the bar
        label = ax.annotate(width,
                            xy=(width, yloc),
                            xytext=(xloc, 0),           # The position (x,y) to place the text at. If None, defaults to xy.
                            textcoords='offset points', # The coordinate system that xytext is given in
                            ha=align, va='center', color='g', weight='bold', clip_on=True);

    for rect in rects2:
        width = rect.get_width()
        width = np.around(width, decimals=2)

        yloc = rect.get_y() + rect.get_height() / 2  # Center the text vertically in the bar
        label = ax.annotate(width,
                            xy=(width, yloc),
                            xytext=(xloc, 0),           # The position (x,y) to place the text at. If None, defaults to xy.
                            textcoords='offset points', # The coordinate system that xytext is given in
                            ha=align, va='center', color='b', weight='bold', clip_on=True);

    # Set xlim
    # x_max = max([max(resp_dict[k]) for k in resp_dict.keys() if k!="labels"]) # find largest value
    # ax.set_xlim([0, x_max + 100])
    
    ax.legend(fontsize=ticks_fontsize);
    return ax


def resp_dist_dict(data, agg_by: str="Samples", var: str="ctype", target: str="Response"):
    """ Generete dict of distribution of responses. """
    aa = data.groupby([var, target]).agg({"smp": "nunique", "Group": "nunique"}).reset_index().rename(
        columns={"smp": "Samples", "Group": "Groups"})
    
    unq_resp = sorted(aa[target].unique())
    resp_dict = {}
    labels = []

    # Iterate over unique values of the var (e.g., ctype), and for unqiue var
    # store the count target (e.g., Response)
    for var_value in aa[var].unique():
        labels.append(var_value)

        for r in unq_resp:
            xx = aa[(aa[var] == var_value) & (aa[target]==r)]
            if xx.shape[0] == 1:
                v = xx[agg_by].tolist()[0]
            elif xx.shape[0] == 0:
                v = 0
            else:
                raise ValueError("something is wrong.")

            if r in resp_dict.keys():
                resp_dict[r].append(v)
            else:
                resp_dict[r] = [v]

    resp_dict["labels"] = labels
    return aa, resp_dict


def main(args):
    # import ipdb; ipdb.set_trace()

    # Load dataframe (annotations)
    dataname = "tidy_drug_pairs_all_samples"
    annotations_file = cfg.DATA_PROCESSED_DIR/dataname/cfg.SF_ANNOTATIONS_FILENAME
    dtype = {"image_id": str, "slide": str}
    data = pd.read_csv(annotations_file, dtype=dtype, engine="c", na_values=["na", "NaN"], low_memory=False)
    print(data.shape)
    
    # By smp
    agg_by = "Samples"
    aa, resp_dict = resp_dist_dict(data, agg_by=agg_by, var="ctype", target="Response")
    ax = barplot_sample_count(resp_dict, title=f"Histogram of {agg_by}", figsize=(8, 6));
    plt.tight_layout()
    ax.set_xlim([0, 2550])

    outpath = fdir/".."/cfg.DATA_PROCESSED_DIR/dataname/"resp_histogram_samples.png"
    plt.savefig(outpath, dpi=150)
    
    # By Group
    agg_by = "Groups"
    aa, resp_dict = resp_dist_dict(data, agg_by=agg_by, var="ctype", target="Response")
    ax = barplot_sample_count(resp_dict, title=f"Histogram of {agg_by}", figsize=(8, 6));
    plt.tight_layout()
    ax.set_xlim([0, 290])

    outpath = fdir/".."/cfg.DATA_PROCESSED_DIR/dataname/"resp_histogram_groups.png"
    plt.savefig(outpath, dpi=150)    
    print("\nDone.")

    
if __name__ == "__main__":
    main(sys.argv[1:])
