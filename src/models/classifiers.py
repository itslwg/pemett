import numpy as np
import pandas as pd

from typing import Optional, Callable

from tqdm.notebook import tqdm
from sklearn.base import clone

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class StackedGeneralizationClassifier():
    """Stacking Generalization Classifier.
    
    Args:
        base_clfs: Base classifiers.
        meta_clf: Meta classifier.
        use_probas: If True, meta classifier is fitted to predicted
            probabilities.
        verbose: If True, logging is used. E.g. when fitting model.
    """
    def __init__(self, base_clfs, meta_clf, 
                 use_probas: bool = True,
                 verbose: bool = False):
        self.base_clfs = base_clfs
        self.meta_clf = meta_clf
        self.use_probas = use_probas
        self.verbose = verbose
        
        # Copy classifiers for fitting
        self.base_clfs_ = base_clfs
        self.meta_clf_ = meta_clf
       
        # Splitting setup
        self.inner_folds = 2
        self.outer_folds = 2
        self.inner_loop = StratifiedKFold(n_splits = self.inner_folds)
        self.outer_loop = StratifiedKFold(n_splits = self.outer_folds)
        
        # AUC of ROC for each outer fold. See self.cv_outer_loop
        self.roc_aucs = None
        
        # Updated with each run. Lastly set to best hyper parameters if refit.
        self.hyper_parameters = None
        
        # Helpers for logging
        self.__i = 0
        self.__j = 0
        self.__n = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object:
        """Fits the classifiers and the meta classifier.
        
        Args:
            X: Features.
            y: Target labels.
        
        Returns:
            The StackedGeneralizationClassifier itself.
        """
        # Set the hyper hyper parameters of the base classifiers
        for clf in self.base_clfs_:
            nm = clf.__class__.__name__.lower()
            ks = [s for s in self.hyper_parameters.keys() if nm in s]
            clf_params = {k.split("__", 1)[1]: self.hyper_parameters.get(k) for k in ks}
            clf.set_params(**clf_params)
        # Get meta features from features
        X_meta = self.cv_inner_loop(X = X, y = y)
        
        # Fit each base classifier to all features
        self.base_clfs_ = [clf.fit(X, y) for clf in self.base_clfs_]

        # Fit meta classifier to meta features of train
        self.meta_clf_.fit(X_meta, y)
    
        return self


    def predict_meta_features(self, X: pd.DataFrame, 
                              use_probas: Optional[bool] = None) -> np.ndarray:
        """Uses base classifiers to get meta features.
        
        Args:
            X: Features.
            use_probas: Optional. If not None and True, predicts probabilities of 
                1s. Else, gives predicted class.
        
        Returns:
            Meta features, i.e. the predicted probabilities by base
            classifiers.
        """
        if use_probas is None: use_probas = self.use_probas

        per_model_predictions = []
        for clf in self.base_clfs_:
            prediction = clf.predict_proba(X)[:, 1] if use_probas else clf.predict(X)
            per_model_predictions.append(prediction[:, np.newaxis])
        
        predictions = np.hstack(per_model_predictions)

        return predictions
        

    def predict(self, X, use_probas: Optional[Callable] = True) -> tuple:
        """Predicts using meta classifier
        
        Args:
            X: Features from which to classify rows.
        
        Returns:
            Tuple of predicted proability of 1s and binned predictions.
        """
        # Predict using validation meta features
        X_meta = self.predict_meta_features(X, use_probas = self.use_probas)
        y_pred = (
            self.meta_clf_.predict_proba(X_meta) if use_probas 
            else self.meta_clf_.predict(X_meta)
        )

        return_object = y_pred
        if use_probas:
            y_pred_cut = pd.cut(
                x=y_pred[:, 1],
                bins=self.hyper_parameters["breaks"],
                labels=[0, 1, 2, 3],
                right=True,
                include_lowest=False
            )
            return_object = (
                y_pred[:, 1],
                np.array(y_pred_cut)
            )

        return return_object

    def cv_inner_loop(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Runs inner loop of k-fold cross-validation.
    
        Args: 
            X: Features.
            y: Target labels.
        Returns:
          Array where each column correspond to the predictions by each
              respective classifier
        """
        if self.__i == 0 and self.__j == 0:
            print("Inner loop, each loop:")
            for k in self.__n.index:
                p = round(self.__n[k] / sum(self.__n) * 100, 2)
                print("\tNumber of {c}'s: ~{v} ({p}%)".format(
                    c=round(k),
                    v=round(self.__n[k] / self.inner_folds),
                    p=p
                ))

        X_meta = []

        for clf in self.base_clfs_:
            if self.verbose: print("Running predictions for " + str(clf))
            predictions = cross_val_predict(
                estimator=clf,
                X=X,
                y=y,
                cv=self.inner_loop,
                method="predict_proba" if self.use_probas else "predict"
            )
            p = predictions[:, 1] if self.use_probas else predictions
            X_meta.append(p)

        return np.column_stack(X_meta)


    def cv_outer_loop(self, all_hyper_parameters: list,
                      X: pd.DataFrame, y: pd.Series, refit=True) -> object:
        """Runs outer cross-validation.
        
        Gets the break points for continous proabilities that 
        yields the greatest AUC of ROC.
        
        Args:
            all_hyper_parameters: Hyper parameters to try for the base
                classifiers
            X: Features.
            y: Targets.
            refit: If True, the base classifiers and the meta classifier
                is as a final step refit to the training set.¨
                
        Returns:
            The StackedGeneralizationClassifier itself.
        """
        ## Setup for recording auc from each combination of hps
        roc_aucs = pd.DataFrame(
            data = np.zeros((len(all_hyper_parameters), self.outer_folds)),
            columns = range(1, self.outer_folds + 1)
        )
        
        for i, hyper_parameters in enumerate(tqdm(all_hyper_parameters)):

            self.__i = i 

            self.hyper_parameters = hyper_parameters

            for j, (train_index, val_index) in enumerate(self.outer_loop.split(X, y)):

                self.__j = j
            
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_val = X.iloc[val_index]
                y_val = y.iloc[val_index]
                
                # Number of each class w. percentage. Only first loop
                if i == 0 and j == 0:
                    n = y_val.value_counts()
                    self.__n = n
                    print("Outer loop, each fold:")
                    for k in n.index:
                        p = round(n[k] / sum(n) * 100, 2)
                        print("\tNumber of {c}'s: ~{v} ({p}%)".format(
                            c=round(k),v=n[k],p=p)
                        )
                
                self.fit(X = X_train, y = y_train)
                y_pred_con, y_pred_cut = self.predict(X=X_val)
    
                auc = roc_auc_score(
                    y_true=y_val,
                    y_score=y_pred_cut
                )
                roc_aucs.iloc[i, j] = auc
        
        self.roc_aucs = roc_aucs
    
        # Find the best performing settings for the models
        max_row = roc_aucs.mean(axis=1).idxmax()
        self.hyper_parameters = all_hyper_parameters[max_row]

        if refit: self.fit(X = X_train, y = y_train)
        
        return self