import numpy as np
import pandas as pd

import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

from tqdm import tqdm
from sklearn.utils import resample
from joblib import Parallel, delayed

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

from src.models.classifiers import StackedGeneralizationClassifier


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


def compute_performance(y_prob: np.array, y_pred: np.array,
                        y_true: np.array, y_pred_cut: np.array = None,
                        tc: pd.Series = None):
    """Computes performance metrics.
    
    For 1 set of predictions.
    
    Args:
        y_prob: Predicted probabilities of 1s.
        y_pred: Predicted class.
        y_true: True labels.
        y_pred_cut: Cut predicted probabilities.
        tc: Clincians predicted triage category.
        
    Returns:
        Dictionary with labelled performance metrics.
    """
    roc_auc = roc_auc_score(
        y_true=y_true, 
        y_score=y_prob
    )
    prec = precision_score(
        y_true=y_true, 
        y_pred=y_pred, 
        average="macro"
    )
    rec = recall_score(
        y_true=y_true, 
        y_pred=y_pred, 
        average="macro"
    )
    r = dict(roc_auc = roc_auc, prec = prec, rec = rec)
    if y_pred_cut is not None:
        auc_metric_labels = ["auc_model_model", "auc_model_tc"]
        # Calculate alternate aucs
        roc_auc_cut = roc_auc_score(
            y_true=y_true, 
            y_score=y_pred_cut
        )        
        roc_auc_tc = roc_auc_score(
            y_true=y_true, 
            y_score=tc
        )
        auc_mm = roc_auc - roc_auc_cut
        auc_mtc = roc_auc - roc_auc_tc
        aucs = [auc_mm, auc_mtc]
        # Calculate nri
        nri_metrics = ["nri", "nri_plus", "nri_minus"]
        nri = calculate_nri(
            y_true=y_true, 
            y_old=tc,
            y_new=y_pred_cut
        )
        r = {
            **r,
            **{ml: m for ml, m in zip(auc_metric_labels, aucs)},
            **{m: nri[m] for m in nri_metrics}
        }

    return r


def compute_metrics(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series,
                    tc: pd.Series, keys: list,
                    base_clfs: list, 
                    meta_clf: callable,
                    all_hyper_parameters: list):
    """Computes relevant performance metrics.
    
    ROC, Precision, Recall, and NRI for the meta classifier.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test targets.
        tc: Clinician priorities for test set.
        keys: Labels for classifiers.
        base_clfs: Base (level 0) classifiers.
        meta_clf: Meta (level 1) classifiers.
        all_hyper_parameters: Hyper parameters to search.
        
    Returns:
        Dictionary with performance metrics, for each classifier.
    """
    # Fit classifier
    sgclf = StackedGeneralizationClassifier(
        base_clfs=base_clfs, 
        meta_clf=meta_clf,
        use_probas=True, 
        verbose=False
    )
    sgclf.cv_outer_loop(
        all_hyper_parameters=all_hyper_parameters,
        X=X_train, 
        y=y_train,
        refit=True
    )
    # Predictions by the meta classifier
    y_test_prob_con, y_test_prob_cut = sgclf.predict(X_test)
    y_test_pred_meta_clf = sgclf.predict(X_test, use_probas = False)
    
    # Predictions for each base classifier
    y_test_prob_clfs = sgclf.predict_meta_features(X_test, use_probas = True)
    y_test_pred_clfs = sgclf.predict_meta_features(X_test, use_probas = False) 

    # Merge the two
    y_preds = np.column_stack((y_test_pred_clfs, y_test_pred_meta_clf))
    y_probs = np.column_stack((y_test_prob_clfs, y_test_prob_con))
    
    # Helper for calculating performance
    y_test_prob_cuts = [None] * (len(keys) - 1) + [y_test_prob_cut]
    ds = {key: compute_performance(
            y_prob=prob, 
            y_pred=pred, 
            y_true=y_test,
            y_pred_cut=cut,
            tc=tc
        ) for prob, pred, key, cut in zip(
            y_probs.T, 
            y_preds.T, 
            keys,
            y_test_prob_cuts
    )}
    
    return ds


def boot_compute_metrics(X: pd.DataFrame, y: pd.Series, 
                         tc: pd.Series, training_size: int,
                         keys: list, base_clfs: list,
                         meta_clf: callable, 
                         all_hyper_parameters: list):
    """Helper to refactor compute_metrics."""
    # Prepare training and test sets
    X_train = resample(X, n_samples=training_size, stratify=y)
    y_train = y.loc[X_train.index]
    tc_train = tc.loc[X_train.index]
    X_test = X.loc[~X.index.isin(X_train.index), :]
    y_test = y.loc[X_test.index]
    tc_test = tc.loc[X_test.index]

    return compute_metrics(
        X_train=X_train,
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        tc=tc_test,
        keys=keys,
        base_clfs=base_clfs,
        meta_clf=meta_clf,
        all_hyper_parameters=all_hyper_parameters
    )


def bootstrap(X: pd.DataFrame, y: pd.Series, tc: pd.Series,
              keys: list, base_clfs: list, meta_clf: callable,
              all_hyper_parameters: list,
              N: int = 5, train_size: float = 0.8, 
              n_jobs: int = 2):
    """Bootstraps statistics
    
    Parallelized computation of bootstrap performance estimates.
    
    Args:
        X: Features.
        y: Targets.
        N: Number of bootstrap samples.
        train_size: Proportion of samples in the training sample.
        n_jobs: Number of parallel processes.
        keys: Classifier keys.
        
    Returns:
        List of estimates from each bootstrap sample.
    """
    # Numbef of samples in training samples
    training_size = int(len(X.index) * train_size)
    # [boot_compute_metrics(X, y, tc, training_size, keys, base_clfs, meta_clf, all_hyper_parameters) for i in tqdm(range(N))]
    return Parallel(n_jobs=n_jobs)(delayed(boot_compute_metrics)(X, y, tc, training_size, keys, base_clfs, meta_clf, all_hyper_parameters) for i in tqdm(range(N)))


def compute_bootstrap_ci(point_estimate, stats):
    """Computes confidence interval.
    
    Uses the empirical bootstrap.
    Source:
        https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        
    Args:
        point_estimate: Point estimate from our sample.
        stats: Estimates from each bootstrap sample.
        
    Returns:
        Tuple of lower and upper bound of confidence interval.
    """
    bs_statistics = np.sort(stats)
    delta_star = bs_statistics - point_estimate
    d1 = np.quantile(delta_star, 0.1)
    d2 = np.quantile(delta_star, 0.9)
    ub = point_estimate - d1
    lb = point_estimate - d2
    
    return lb, ub
