from plotnine import *

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)


def plot_roc(y_true, y_pred):
    """Plots ROC using plotnine."""
    r = roc_curve(y_true = y_true, y_score = 1 - y_pred)
    plot_data = pd.DataFrame(
        np.column_stack(r),
        columns=["tpr", "fpr", "thresholds"]
    )
    p = (
        ggplot(plot_data, aes(x="fpr", y="tpr")) +
        geom_line() + 
        xlab("False Positive Rate") + 
        ylab("True Positive Rate")
    )
    return p


def plot_roc_base_clfs(y_preds: np.ndarray, y_true: np.array, keys: list):
    """Plots ROC for each base classifier."""
    def f(p, y_true):
        fpr, tpr, _ = roc_curve(y_true = y_true, y_score = p)
        a = np.column_stack((fpr, tpr))
        df = pd.DataFrame(a, columns=["fpr", "tpr"])
        return df

    df = pd.DataFrame(y_preds)
    dfs = [f(df[l], y_true = y_true) for l in df.columns]
    df_plt = pd.concat(dfs, keys=keys).reset_index(level=0)
    plt = (
        ggplot(df_plt, aes(x = "fpr",y = "tpr", color = "level_0")) +
        geom_line() + 
        xlab("False Positive Rate") + 
        ylab("True Positive Rate") +
        theme(legend_title=element_blank())
    )

    return plt


def plot_precision_recall(y_true, y_pred):
    """Plots precision and recall using plotnine."""
    prec, rec, thres = precision_recall_curve(y_true = y_true, probas_pred = 1 - y_pred)
    plot_data = pd.DataFrame(
        np.column_stack((prec, rec)),
        columns=["precision", "recall"]
    )
    p = (
        ggplot(plot_data, aes(x="precision", y="recall")) +
        geom_line() + 
        xlab("Precision") + 
        ylab("Recall")
    )
    return p


    