import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.model_selection import cross_val_predict, StratifiedKFold


class StackingGeneralization():
    """Stacking Generalization Classifier.
    
    Args:
        base_clfs: Base classifiers.
        meta_clf: Meta classifier.
        verbose: If True, logging is used. E.g. when fitting model.
    """
    def __init__(self, base_clfs, meta_clf, verbose: bool = False):
        self.base_clfs = base_clfs
        self.meta_clf = meta_clf
        self.verbose = verbose
        
        # Final fitted classifiers
        self.base_clfs__ = []
        self.meta_clf__= meta_clf
       
        # Splitting setup
        self.inner_folds = 3
        self.outer_folds = 2
        self.inner_loop = StratifiedKFold(n_splits = self.inner_folds)
        self.outer_loop = StratifiedKFold(n_splits = self.outer_folds)
        
        # AUC of ROC for each outer fold. See self.cv_outer_loop
        self.roc_aucs = None
        
        # Updated with each run. Lastly set to best hyper parameters if refit.
        self.hyper_parameters = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the classifiers and the meta classifier.
        
        Args:
            X: Features.
            y: Target labels.
        
        Returns:
            Returns the classifier itself.
        """
        # Set the hyper hyper parameters of the base classifiers
        for clfk in self.base_clfs.keys():
            ks = [s for s in self.hyper_parameters.keys() if clfk in s]
            clf_params = {k.split("__")[1]: self.hyper_parameters.get(k) for k in ks}
            clf = self.base_clfs[clfk]
            clf.set_params(**clf_params)
            self.base_clfs__.append(clf)
    
        # Get meta features from features
        X_meta = self.cv_inner_loop(X = X, y = y)

        # Fit meta classifier to meta features of train
        self.meta_clf__.fit(X_meta, y)
    
        return self


    def predict_meta_features(self, X):
        """Uses base classifiers to get meta features.
        
        Args:
            X: Features.
        
        Returns:
            Meta features, i.e. the predicted probabilities by base
            classifiers.
        
        """
        per_model_predictions = []
        for clf in self.base_clfs__:
            prediction = clf.predict_proba(X)[:, 1]
            per_model_predictions.append(prediction[:, np.newaxis])
        
        predictions = np.hstack(per_model_predictions)
        return predictions
        

    def predict(self, X) -> tuple:
        """Predicts using meta classifier
        
        Args:
            X: Features from which to classify rows.
        
        Returns:
            Tuple of predicted proability of 1s and binned predictions.
        """
        # Predict using validation meta features
        y_pred_con = self.meta_clf__.predict_proba(X)
        # Calculate AUC of ROC for cut predictions
        y_pred_cut = pd.cut(
            x=y_pred_con[:, 1],
            bins=self.hyper_parameters["breaks"],
            labels=[0, 1, 2, 3],
            right=True,
            include_lowest=False
        )

        return y_pred_con[:, 1], np.array(y_pred_cut)

    def cv_inner_loop(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Runs inner loop of k-fold cross-validation.
    
        Args: 
            X: Features.
            y: Target labels.
        Returns:
          Each column represent predictions by each respective classifier
        """
        X_meta = np.zeros((len(X.index), ))
        for clf in self.base_clfs__:
            if self.verbose: print("Running predictions for " + str(clf))
            predictions = cross_val_predict(
                estimator=clf,
                X=X,
                y=y,
                cv=self.inner_loop
            )
            if X_meta.any():
                X_meta = np.hstack([X_meta, predictions[:, np.newaxis]])
            else:
                X_meta = predictions[:, np.newaxis]
    
        return X_meta


    def cv_outer_loop(self, all_hyper_parameters: list,
                      X: pd.DataFrame, y: pd.Series, refit=True):
        """Runs outer cross-validation to get best break points."""
        ## Setup for recording auc from each combination of hps
        roc_aucs = pd.DataFrame(
            data = np.zeros((len(all_hyper_parameters), self.outer_folds)),
            columns = range(1, self.outer_folds + 1)
        )
        
        for i, hyper_parameters in enumerate(tqdm(all_hyper_parameters)):

            self.hyper_parameters = hyper_parameters

            for j, (train_index, val_index) in enumerate(self.outer_loop.split(X, y)):
            
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_val = X.iloc[val_index]
                y_val = y.iloc[val_index]
                
                self.fit(X = X_train, y = y_train)
                
                X_val_meta = self.predict_meta_features(X = X_val)
                
                y_pred_con, y_pred_cut = self.predict(X=X_val_meta)
    
                auc = roc_auc_score(
                    y_true=y_val,
                    y_score=y_pred_cut
                )
                roc_aucs.iloc[i, j] = 1 - auc
        
        self.roc_aucs = roc_aucs
    
        # Find the best performing settings for the models
        max_row = roc_aucs.mean(axis=1).idxmax()
        self.hyper_parameters = all_hyper_parameters[max_row]

        if refit: self.fit(X = X_train, y = y_train)
        
        return self