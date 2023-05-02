import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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


def load_data() -> Dataset:
    df = pd.read_csv("velo.csv")

    Xy = df.values

    X = Xy[:, :-1]
    y = Xy[:, -1]

    train_size = 120

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return (X_train, y_train), (X_test, y_test)


def get_model_parameters(model: LinearRegression) -> LogRegParams:

    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]

    return params


def set_model_params(model: LinearRegression, params: LogRegParams) -> LinearRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

# set initial parameters for Linear Regression
def set_initial_params(model: LinearRegression):
    n_features = 3  # there are three independent variables 
    model.coef_ = np.zeros((1, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1, 1))
