import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression


class PretrainedStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacking classifier that works with already-trained base models.
    Fits only the final meta-learner.
    """
    def __init__(self, estimators, final_estimator=LogisticRegression(), use_probas=False):
        """
        Parameters
        ----------
        estimators : list of (str, estimator) tuples
            List of (name, fitted_model) pairs.
        final_estimator : estimator
            Meta-learner to combine base models' predictions.
        use_probas : bool, default=False
            If True, stack predicted probabilities instead of class labels.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.use_probas = use_probas

        # Store already-trained estimators
        self.estimators_ = [est for _, est in estimators]
        self.named_estimators_ = dict(estimators)

    def _get_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        if self.use_probas:
            meta_features = np.hstack([
                est.predict_proba(X) for est in self.estimators_
            ])
        else:
            meta_features = np.column_stack([
                est.predict(X) for est in self.estimators_
            ])
        return meta_features

    def fit(self, X_meta, y):
        """
        Fit only the final estimator using meta-features.
        The base models are assumed to be pre-trained.
        """
        meta_features = self._get_meta_features(X_meta)
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(meta_features, y)
        return self

    def predict(self, X):
        """Predict using the trained meta-learner."""
        check_is_fitted(self, "final_estimator_")
        meta_features = self._get_meta_features(X)
        return self.final_estimator_.predict(meta_features)

    def predict_proba(self, X):
        """Predict probabilities using the trained meta-learner."""
        check_is_fitted(self, "final_estimator_")
        meta_features = self._get_meta_features(X)
        if hasattr(self.final_estimator_, "predict_proba"):
            return self.final_estimator_.predict_proba(meta_features)
        else:
            raise AttributeError("Final estimator does not support predict_proba.")

    def score(self, X, y):
        """Return accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))