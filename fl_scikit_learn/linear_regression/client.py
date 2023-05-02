import warnings
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import numpy as np
import flwr as fl

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import utils


if __name__ == "__main__":
    # load MNIST dataset
    (X_train, y_train), (X_test, y_test) = utils.load_data()

    # split train set into 10 partitions and randomly use one for training
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # create LinearRegression model
    model = LinearRegression()

    # setting initial parameters
    utils.set_initial_params(model)

    # Define flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model, parameters)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            print(f"Traning finished for round {config['server_round']}")

            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):
            utils.set_model_params(model, parameters)
            loss = mean_squared_error(y_test, model.predict(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower Client
    fl.client.start_numpy_client(server_address="localhost:8000", client=MnistClient())