import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.bounds = {}

        for col in self.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.bounds[col] = (lower, upper)

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            lower, upper = self.bounds[col]
            X[col] = np.clip(X[col], lower, upper)

        return X