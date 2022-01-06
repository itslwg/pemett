import numpy as np
import pandas as pd

import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects


def calculate_nri(y_true: np.array, y_old: np.array, y_new: np.array) -> pd.Series:
    """Calculates Net Reclassification Improvement (NRI)
    
    Uses the R package nricens to do so.
    
    Args:
        y_true: Actual target values.
        y_old: Model predictions by old classifier.
        y_new: Model predictions by new classifier.
        
    Returns:
        Series with nri components and information on movement
        in categories.
    """
    nricens = rpackages.importr("nricens")
    utils = rpackages.importr("utils")

    rnri = nricens.nribin(
        event = robjects.vectors.IntVector(y_true.tolist()), 
        p_std = robjects.vectors.IntVector(y_old.tolist()), 
        p_new = robjects.vectors.IntVector(y_new.tolist()),
        cut = [2, 3, 4],
        niter = 0,
        msg = False
    ).rx2("nri")

    labels = [
        "nri", "nri_plus", 
        "nri_minus", "pr_up_dead", 
        "pr_down_dead", "pr_down_surv", 
        "pr_up_surv"
    ]
    nri = pd.Series(np.asarray(rnri).flatten(), index = labels)

    return nri