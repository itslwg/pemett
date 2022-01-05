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

    