from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x=np.linspace(-1.2,2,n_samples)

    epsilon = np.random.normal(0,noise,n_samples)
    y = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    y_noise=y+epsilon
    train_x,train_y,test_x,test_y=split_train_test(pd.DataFrame(x),pd.Series(y_noise),2/3)

    plt.title(f"The average training and validation errors with:\n {n_samples} samples and noise={noise}")
    plt.scatter(x, y, c='blue')
    plt.scatter(train_x, train_y, c='green')
    plt.scatter(test_x, test_y, c='red')
    plt.legend(["Noiseless" , "Train", "Test"], ncol = 3 , loc = "upper right")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    scores=[]
    for k in range(11):
        polynomial_model=PolynomialFitting(k)
        curr_train_score, curr_validation_score = cross_validate(polynomial_model, train_x.to_numpy(), train_y.to_numpy(), mean_square_error)
        scores.append({"train_score":curr_train_score, "validation_score":curr_validation_score, "k":k})
    scores_dataframe = pd.DataFrame(scores)

    plt.title(f"The noiseless model and the the train and test sets with:\n {n_samples} samples and noise={noise}")
    plt.plot(scores_dataframe["k"], scores_dataframe["train_score"],'-p', c='blue')
    plt.plot(scores_dataframe["k"], scores_dataframe["validation_score"],'-p', c='green')
    plt.legend(["Avg training score" , "Avg validation score"], ncol = 2 , loc = "upper right")
    plt.xlabel("degree")
    plt.ylabel("avg error")
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_val_score=scores_dataframe["validation_score"].min()
    best_model=scores_dataframe[scores_dataframe["validation_score"]==best_val_score]
    best_k=int(best_model["k"])
    best_polynomial_model=PolynomialFitting(best_k)
    best_polynomial_model.fit(train_x.to_numpy(), train_y.to_numpy())
    error=round(best_polynomial_model.loss(test_x.to_numpy(), test_y.to_numpy()),2)
    print(f"Number of samples: {n_samples}, noise= {noise}")
    print(f"Best k={best_k}, Test error is: {error}\n")


def calc_test_regularization(estimator, X, y, lambda_arr):
    train_score_arr=np.zeros(len(lambda_arr))
    validation_score_arr=np.zeros(len(lambda_arr))

    for i in range(len(lambda_arr)):
        curr_estimator=estimator(lambda_arr[i])

        train_score_arr[i], validation_score_arr[i]=cross_validate(curr_estimator, X,y,mean_square_error)
    return train_score_arr, validation_score_arr

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y=datasets.load_diabetes(return_X_y=True)

    train_proportion = n_samples / X.shape[0]
    train_x, train_y, test_x,test_y = split_train_test(pd.DataFrame(X),pd.Series(y),train_proportion)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    fig, ax = plt.subplots()
    lambda_arr = np.linspace(10**-5, 6, n_evaluations)#TODO:changeeeeeeeeeeee
    train_score_ridge,validation_score_ridge=calc_test_regularization(RidgeRegression,train_x.to_numpy(),train_y.to_numpy(),lambda_arr)
    train_score_lasso,validation_score_lasso=calc_test_regularization(Lasso, train_x.to_numpy(), train_y.to_numpy(), lambda_arr)
    ax.plot(lambda_arr, train_score_ridge, c='blue')
    ax.plot(lambda_arr, validation_score_ridge,c='red')
    ax.plot(lambda_arr, train_score_lasso,c='green')
    ax.plot(lambda_arr, validation_score_lasso,c='pink')
    ax.set_xlabel('lambda')
    ax.set_ylabel('error')#TODO
    plt.legend(["Avg training score Ridge" , "Avg validation score Ridge","Avg training score Lasso" , "Avg validation score Lasso"], ncol =2)
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_val_score_ridge_ind = np.argmin(validation_score_ridge)
    best_val_score_lasso_ind = np.argmin(validation_score_lasso)
    best_ridge_model=RidgeRegression(lambda_arr[best_val_score_ridge_ind])
    best_ridge_model.fit(train_x.to_numpy(),train_y.to_numpy())

    best_lasso_model = Lasso(lambda_arr[best_val_score_lasso_ind])
    best_lasso_model.fit(train_x.to_numpy(), train_y.to_numpy())

    linear_reg_model=LinearRegression()
    linear_reg_model.fit(train_x.to_numpy(), train_y.to_numpy())

    print("Best lambda of Ridge: ", lambda_arr[best_val_score_ridge_ind])
    print("Ridge best error: ",best_ridge_model.loss(test_x.to_numpy(),test_y.to_numpy()))

    print("Best lambda of lasso: ", lambda_arr[best_val_score_lasso_ind])
    print("Lasso best error: ", mean_square_error(test_y.to_numpy(), best_lasso_model.predict(test_x)))

    print("Linear Regression error: ", linear_reg_model.loss(test_x.to_numpy(), test_y.to_numpy()))


if __name__ == '__main__':
    np.random.seed(0)
    # Questions 1-3
    select_polynomial_degree()
    #Question 4
    select_polynomial_degree(noise=0)
    #Question 5
    select_polynomial_degree(n_samples=1500, noise=10)
    #Question 6
    select_regularization_parameter()
