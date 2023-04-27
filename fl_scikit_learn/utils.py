import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from typing import Tuple, Union, List

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    return list(
        zip(
            np.array_split(X, num_partitions),
            np.array_split(y, num_partitions)
        )
    )


def load_mnist() -> Dataset:
    df = pd.read_csv("MNIST.csv")
    df = df.drop(['Unnamed: 0'], axis=1)
    
    Xy = df.values

    X = Xy[:, :-1]
    y = Xy[:, -1]

    train_size = 60000

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return (X_train, y_train), (X_test, y_test)


def get_model_parameters(model: LogisticRegression) -> LogRegParams:

    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]

    return params


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    n_classes = 10
    n_features = 784
    model.classes_ = np.arange(0, 10, 1)
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes, ))
