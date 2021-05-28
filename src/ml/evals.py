from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


def calc_preds(model, x, y, mltype):
    """ Calc predictions. """
    if mltype == 'cls': 
        def get_pred_fn(model):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba
            if hasattr(model, 'predict'):
                return model.predict

        pred_fn = get_pred_fn(model)
        if (y.ndim > 1) and (y.shape[1] > 1):
            y_pred = pred_fn(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_pred = pred_fn(x)
            y_true = y
            
    elif mltype == 'reg':
        y_pred = np.squeeze(model.predict(x))
        y_true = np.squeeze(y)

    return y_pred, y_true


def dump_preds(y_true, y_pred, meta=None, outpath='./preds.csv'):
    """ Dump prediction and true values, with optional with metadata. """
    y_true = pd.Series(y_true, name='y_true')
    y_pred = pd.Series(y_pred, name='y_pred')
    if meta is not None:
        # preds = meta.copy()
        # preds.insert(loc=3, column='y_true', value=y_true.values)
        # preds.insert(loc=4, column='y_pred', value=y_pred.values)
        preds = pd.concat([meta, y_true, y_pred], axis=1)
    else:
        preds = pd.concat([y_true, y_pred], axis=1)
    preds.to_csv(Path(outpath), index=False)
    return None


def calc_scores(y_true, y_pred, mltype, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    scores = {}

    if mltype == "cls":    
        # Metric that accept probabilities
        scores["brier"] = sklearn.metrics.brier_score_loss(y_true, y_pred, sample_weight=None, pos_label=1)
        scores["roc_auc"] = sklearn.metrics.roc_auc_score(y_true, y_pred)

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        scores["pr_auc"] = sklearn.metrics.auc(recall, precision)

        # Metric that don't accept probabilities
        y_pred_ = [0 if v < 0.5 else 1 for v in y_pred]
        scores["mcc"] = sklearn.metrics.matthews_corrcoef(y_true, y_pred_, sample_weight=None)
        scores["f1_score"] = sklearn.metrics.f1_score(y_true, y_pred_, average="binary")
        scores["acc_blnc"] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred_)
        scores["recall"] = sklearn.metrics.recall_score(y_true, y_pred_)
        scores["precision"] = sklearn.metrics.precision_score(y_true, y_pred_)

    elif mltype == "reg":
        scores["r2"] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores["mean_absolute_error"]   = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores["median_absolute_error"] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        scores["mse"]  = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores["rmse"] = scores["mse"] ** 0.5
        # scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_pred)
        scores["spearmanr"] = spearmanr(y_true, y_pred)[0]
        scores["pearsonr"] = pearsonr(y_true, y_pred)[0]
        
        scores["y_mean_true"] = np.mean(y_true)
        scores["y_mean_pred"] = np.mean(y_pred)
        
    # # https://scikit-learn.org/stable/modules/model_evaluation.html
    # for metric_name, metric in metrics.items():
    #     if isinstance(metric, str):
    #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
    #         scores[metric_name] = scorer(ydata, pred)
    #     else:
    #         scores[metric_name] = scorer(ydata, pred)    
    return scores


def save_confusion_matrix(true_labels, predictions, p=0.5,
                          labels=["Non-response", "Response"],
                          outpath="confusion.jpeg", figsize=(3, 3)):
    """ ... """
    cnf_mtrx = confusion_matrix(true_labels, predictions > p)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cnf_mtrx, annot=True, fmt="d", cmap="Blues", linewidths=0.2, linecolor="white")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True", xlabel="Predicted")
    ax.set_title("Confusion matrix at p={:.2f}".format(p))
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    return None


def plot_calibration_curve(name, fig_index, y_true, probs, outpath):
    """ Plot calibration curve for est w/o and with calibration.
    https://towardsdatascience.com/classifier-calibration-7d0be1e05452
    """
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2)
    ax2 = plt.subplot2grid(shape=(3, 1), loc=(2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    frac_of_pos, mean_pred_value = calibration_curve(y_true, probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f"{name}")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="best")
    ax1.set_title(f"Calibration plot ({name})")

    ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    return None


# def scores_to_df(scores_all):
#     """ Dict to df. """
#     df = pd.DataFrame(scores_all)
#     df = df.melt(id_vars=['run'])
#     df = df.rename(columns={'variable': 'metric'})
#     df = df.pivot_table(index=['run'], columns=['metric'], values='value')
#     df = df.reset_index(drop=False)
#     df.columns.name = None
#     return df
