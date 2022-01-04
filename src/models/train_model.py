import itertools

import pandas as pd
import numpy as np
from typing import Callable


def generate_all_combinations(d):
    """All permutations of dict elements.
    
    Source:
        https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    """
    keys, values = zip(*d.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def fit(base_clfs: list, meta_clf: Callable,
        inner_loop: Callable, hyper_parameters: list, 
        X_train: pd.DataFrame, y_train: pd.Series, 
        verbose: bool):
    """Fits the classifiers and the meta classifier.
    
    Fits all the base classifiers, get predictions on all inner
    loop validaion folds, and fit the meta classifiers to the predicted
    probabilities.
    
    Args:
        base_clfs: Base classifiers
        meta_clf: Meta classifier
        hyper_parameters: Hyper parameters for base classifiers and
            breaks for binning continous predictions.
        X_train: Training features.
        y_train: Training targets.
        verbose: If True, logging is printed in inner cross-validation.
    
    Returns:
        Tuple of predicted proability of 1s and binned predictions.
    """
    base_clfs_ = []
    # Set the hyper hyper parameters of the base classifiers
    for clfk in base_clfs.keys():
        ks = [s for s in hyper_parameters.keys() if clfk in s]
        clf_params = {k.split("__")[1]: hyper_parameters.get(k) for k in ks}
        clf = base_clfs[clfk]
        clf.set_params(**clf_params)
        base_clfs_.append(clf)
    
    # Get meta features of training set
    meta_features_train = cv_inner_loop(
        base_clfs=base_clfs_,
        inner_loop=inner_loop,
        X=X_train,
        y=y_train,
        verbose=verbose
    )

    # Fit meta classifier to meta features of train
    meta_clf.fit(meta_features_train, y_train)
    
    return base_clfs_, meta_clf, meta_features_train


def cv_inner_loop(base_clfs: list,
                  inner_loop: Callable, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  verbose: bool) -> np.ndarray:
    """Run inner loop of k-fold cross-validation.
    
    Uses sklearn's cross_val_predict.
    
    That is,
    1. Fit classifier to the training folds.
    2. Make prediction on the validation fold.
    3. Use all folds as validation fold, one time each.

    Args:
      base_clfs: List of classifiers. E.g. [LGBMClassifier, LogisticRegression]
      inner_loop: scikit-learn callable to split into folds
      X: Features
      y: Targets

    Returns:
      Each column represent predictions by each respective classifier
    """
    predictions = np.zeros((len(X_train.index), ))
    for clf in base_clfs:
        if verbose: print("Running predictions for " + str(clf))
        preds = cross_val_predict(
            estimator=clf,
            X=X,
            y=y,
            cv=inner_loop
        )
        if predictions.any():
            predictions = np.hstack([predictions, preds[:, np.newaxis]])
        else:
            predictions = preds[:, np.newaxis]

    return predictions


def cv_outer_loop(base_clfs: list, meta_clf: Callable,
                  all_hyper_parameters: list,
                  X: pd.DataFrame, y: pd.Series,
                  use_meta_features: bool = False, 
                  verbose: bool = False):
    """Runs outer cross-validation.
    
    That is, find the best combination cut-points for the classifier.
    "Best" is defined by the highest AUC of ROC.
    
    Inspired by:
        https://github.com/rasbt/mlxtend/blob/master/mlxtend/classifier/stacking_cv_classification.py.
        
    Args:
        base_clfs: Base classifiers.
        meta_clf: Meta classifier.
        all_hyper_parameters: Model hyper parameters and breaks for continous probabilities.
        X: Features.
        y: Targets.
        use_meta_features: If True, the feature set for meta classifier is predicted probabilities
            of positive labels from base classifiers + features used to train
            base classifiers.
        verbose: If True, logging is used in the inner cross-validation.

    Returns:
        Three variables
        - Refitted base classifiers,
        - Refitted meta classifier, and 
        - The hyper parameters yielding the best results.
    """                
    ## Setup splitting
    inner_folds = 3
    outer_folds = 2
    inner_loop = StratifiedKFold(n_splits = inner_folds)
    outer_loop = StratifiedKFold(n_splits = outer_folds)
    
    ## Setup for recording auc from each combination of hps
    roc_aucs = pd.DataFrame(
        data = np.zeros((len(all_hyper_parameters), outer_folds)),
        columns = range(1, outer_folds + 1)
    )
    
    for i, hyper_parameters in enumerate(all_hyper_parameters):

        for j, (train_index, val_index) in enumerate(outer_loop.split(X, y)):
        
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]
            
            y_pred_con, y_pred_cut = predict(
                base_clfs=base_clfs,
                meta_clf=meta_clf,
                hyper_parameters=hyper_parameters,
                inner_loop=inner_loop,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                verbose=verbose
            )
            auc = roc_auc_score(
                y_true=y_val,
                y_score=y_pred_cut
            )
            roc_aucs.iloc[i, j] = 1 - auc

    # Find the best performing settings for the models
    max_row = roc_aucs.mean(axis=1).idxmax()
    best_hyper_parameters = all_hyper_parameters[max_row]
    
    base_clfs_, meta_clf_ = base_clfs, meta_clf
    # if refit: 
    #     base_clfs_, meta_clf_, _ = fit(
    #         base_clfs=base_clfs,
    #         meta_clf=meta_clf,
    #         inner_loop=inner_loop,
    #         hyper_parameters=hyper_parameters,
    #         X_train=X_train,
    #         y_train=y_train,
    #         verbose=verbose
    #     )
    
    return base_clfs_, meta_clf_, best_hyper_parameters


def predict(base_clfs: list, meta_clf: Callable,
            inner_loop: Callable, hyper_parameters: list,
            X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, verbose: bool = False) -> tuple:
    """Predicts using meta classifier.
    
    Use the fitted base classifiers to predict on the validation set,
    fit the meta classifier to those predictions, and predict with 
    the meta classifier.
    
    Args:
        base_clfs: Base classifiers.
        meta_clf: Meta classifier.
        hyper_parameters: Hyper parameters for base classifiers and
            breaks for binning continous predictions.
        inner_loop: Scikit-learn cross-validator. E.g. StratifiedKFold.
        X_train: Training features,
        y_train: Training targets.
        X_val: Validation features.
    
    Returns:
        Tuple of predicted proability of 1s and binned predictions.
    """
    base_clfs_, meta_clf_, meta_features_train = fit(
        base_clfs=base_clfs,
        meta_clf=meta_clf,
        inner_loop=inner_loop,
        hyper_parameters=hyper_parameters,
        X_train=X_train,
        y_train=y_train,
        verbose=verbose
    )

    # Get meta features of validation set
    per_model_preds = []
    for clf in base_clfs_:
        clf.fit(X_train, y_train)
        prediction = clf.predict_proba(X_val)[:, :-1]
        per_model_preds.append(prediction)
    meta_features_val = np.hstack(per_model_preds)
    
    # Fit meta classifier to meta features of train
    meta_clf_.fit(meta_features_train, y_train)
    # Predict using validation meta features
    y_pred_con = meta_clf.predict_proba(meta_features_val)
    # Calculate AUC of ROC for cut predictions
    y_pred_cut = pd.cut(
        x=y_pred_con[:, 1],
        bins=hyper_parameters["breaks"],
        labels=[0, 1, 2, 3],
        right=True,
        include_lowest=False
    )
    return y_pred_con[:, 1], y_pred_cut

