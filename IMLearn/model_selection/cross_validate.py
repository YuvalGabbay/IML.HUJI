from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_folds = np.array_split(X,cv)
    y_folds = np.array_split(y, cv)
    train_score_arr=[]
    validation_score_arr=[]
    for k in range(len(y_folds)):
        curr_x_fold=x_folds[k]
        curr_y_fold = y_folds[k]

        copy_x_folds=deepcopy(x_folds)
        copy_x_folds.pop(k)
        curr_data=np.concatenate(copy_x_folds)

        copy_y_folds = deepcopy(y_folds)
        copy_y_folds.pop(k)
        curr_labels = np.concatenate(copy_y_folds)

        estimator.fit(curr_data, curr_labels)
        curr_train_score = scoring(estimator.predict(curr_data), curr_labels)
        train_score_arr.append(curr_train_score)

        curr_valid_score = scoring(estimator.predict(curr_x_fold), curr_y_fold)
        validation_score_arr.append(curr_valid_score)

    return np.float(np.mean(np.array(train_score_arr))), np.float(np.mean(np.array(validation_score_arr)))


