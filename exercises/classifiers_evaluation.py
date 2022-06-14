from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    X_arr, y_arr=[],[]
    def callback_function(fit: Perceptron, x: np.ndarray, y: int):
        curr_loss=fit.loss(np.array(X_arr),np.array(y_arr))
        losses.append(curr_loss)

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        losses = []
        X_arr, y_arr = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron(callback=callback_function)
        perceptron.fit(X_arr, y_arr)

        # Plot figure of loss as function of fitting iteration
        title=f"Plot fit progression of the Perceptron algorithm over {n} dataset"#TODO:chane title
        fig = px.line(x=range(len(losses)), y=losses, title=title)
        fig.update_layout(xaxis_title="iteration", yaxis_title="loss value")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        x,y_true=load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda=LDA()
        lda.fit(x,y_true)
        naive=GaussianNaiveBayes()
        naive.fit(x,y_true)
        y_pred_lda=lda.predict(x)
        y_pred_naive=naive.predict(x)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy=accuracy(y_true,y_pred_lda)
        naive_accuracy=accuracy(y_true,y_pred_naive)
        lda_title=f"LDA from {f}, accuracy={lda_accuracy}"
        naive_title=f"Gaussian NaiveBayes from {f}, accuracy={naive_accuracy}"
        fig = make_subplots(rows=1, cols=2,subplot_titles=(naive_title, lda_title))


        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=x[:,0], y=x[:,1],mode="markers",
                          marker=dict(color=y_pred_naive, symbol=y_true)),row=1, col=1)

        fig.add_trace(
            go.Scatter(x=x[:,0], y=x[:,1],mode="markers",
                          marker=dict(color=y_pred_lda, symbol=y_true)),row=1, col=2)


        fig.update_layout(title_text="Gaussian Naive Bayes vs LDA prediction scatter plots")


        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in lda.classes_:
            for j in [1,2]:
                if j==1:
                    x_axis=naive.mu_[i][0]
                    y_axis = naive.mu_[i][1]
                    mu_i=naive.mu_[i]
                    cov = np.zeros((x.shape[1], x.shape[1]))
                    np.fill_diagonal(cov,naive.vars_[i])

                else:
                    x_axis = lda.mu_[i][0]
                    y_axis = lda.mu_[i][1]
                    mu_i=lda.mu_[i]
                    cov = lda.cov_

                fig.add_trace(get_ellipse(mu_i,cov), row=1, col=j)
                # Add `X` dots specifying fitted Gaussians' means
                fig.add_trace(go.Scatter(x=[x_axis], y=[y_axis],mode='markers',
                                                marker=dict(size=16, symbol="x", color="black"),
                                                line=dict(width=5)), row=1, col=j)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()


