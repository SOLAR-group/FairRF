import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class PretrainedVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn-style VotingClassifier that works with already-trained models.
    Compatible API: estimators_, named_estimators_, voting, predict, predict_proba, score.
    """

    def __init__(self, estimators, voting='hard'):
        """
        Parameters
        ----------
        estimators : list of (str, estimator) tuples
            List of (name, fitted_model) pairs.
        voting : str, 'hard' or 'soft'
            Voting type:
                - 'hard' : majority vote
                - 'soft' : average predicted probabilities
        """
        self.estimators = estimators
        self.voting = voting

        # These are set immediately instead of during fit()
        self.estimators_ = [est for _, est in estimators]
        self.named_estimators_ = dict(estimators)

    def fit(self, X=None, y=None):
        """
        Does nothing — assumes estimators are already trained.
        Included only for API compatibility.
        """
        return self

    def predict(self, X):
        """Predict class labels for X."""
        check_is_fitted(self, "estimators_")
        if self.voting == 'soft':
            return np.argmax(self.predict_proba(X), axis=1)
        elif self.voting == 'hard':
            predictions = np.asarray([est.predict(X) for est in self.estimators_]).T
            predictions = predictions.astype(int) 
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=predictions
            )
        else:
            raise ValueError("Voting must be 'hard' or 'soft'.")

    def predict_proba(self, X):
        """Predict class probabilities for X (only for soft voting)."""
        if self.voting != 'soft':
            raise AttributeError("predict_proba is only available when voting='soft'.")
        check_is_fitted(self, "estimators_")
        probas = [est.predict_proba(X) for est in self.estimators_]
        return np.average(probas, axis=0)

    def score(self, X, y):
        """Return accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))