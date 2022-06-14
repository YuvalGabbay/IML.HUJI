from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        unique_arr, count = np.unique(y, return_counts=True)
        self.classes_ = unique_arr
        self.mu_ = np.ndarray((len(self.classes_), (X.shape[1])))
        self.vars_ = np.zeros((len(self.classes_), (X.shape[1])))
        for i in self.classes_:
            curr_mean = np.mean(X[y == i], axis=0)
            self.mu_[i] = curr_mean
            self.vars_[i]=np.var(X[y==i],axis=0, ddof=1)

        self.pi_ = count / len(y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        responses = np.argmax(likelihood, axis=1)
        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood=np.zeros((len(X),len(self.classes_)))
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        for i in range(len(X)):
            for k in self.classes_:
                np.fill_diagonal(cov_k,self.vars_[k])
                cov_k_inv=np.linalg.inv(cov_k)
                scalar = 1 / np.sqrt(pow((2 * np.pi), X.shape[1]) * np.linalg.det(cov_k))
                exp = np.exp(-0.5*(X[i]-self.mu_[k])@cov_k_inv@(X[i]-self.mu_[k]))

                likelihood[i][k]=scalar*exp*self.pi_[k]
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

