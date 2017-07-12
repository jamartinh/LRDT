from .dt_transformer import *

__all__ = ['LRDTClassifier']
import pandas as pd


class LRDTClassifier(DTRTransformer):
    """
    A classifier version of the DTRTransformer
    """

    def fit(self, X, y, sample, sample_weight = None):
        if not hasattr(X, 'columns'):
            new_X = pd.DataFrame(X)
            new_X.columns = ['x' + str(c) for c in list(new_X.columns)]
            self.fit_tree(new_X, y)
            X_t = self.transform(new_X)
        else:
            self.fit_tree(X, y)
            X_t = self.transform(X)
        if self.verbose == 2:
            print("Number of rules:", len(self.rule_dict))
        return self.estimator.fit(X_t, y)

    def predict(self, X):
        if not hasattr(X, 'columns'):
            new_X = pd.DataFrame(X)
            new_X.columns = ['x' + str(c) for c in list(new_X.columns)]
            X_t = self.transform(new_X)
        else:
            X_t = self.transform(X)

        return self.estimator.predict(X_t)

    def fit_predict(self, X, y = None):
        self.fit_tree(X, y)
        X_t = self.transform(X)
        self.estimator.fit(X_t, y)
        y_pred_class = self.estimator.predict(X_t)
        return y_pred_class

    def predict_proba(self, X):
        if not hasattr(X, 'columns'):
            new_X = pd.DataFrame(X)
            new_X.columns = ['x' + str(c) for c in list(new_X.columns)]
            X_t = self.transform(new_X)
        else:
            X_t = self.transform(X)

        return self.estimator.predict_proba(X_t)

    def score(self, X, y, sample_weight = None):
        if not hasattr(X, 'columns'):
            new_X = pd.DataFrame(X)
            new_X.columns = ['x' + str(c) for c in list(new_X.columns)]
            X_t = self.transform(new_X)
        else:
            X_t = self.transform(X)

        return self.estimator.score(X_t, y, sample_weight)
