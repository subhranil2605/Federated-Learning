import flwr as fl
import utils
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from typing import Dict


def fit_round(server_round: int) -> Dict:
    return {"server_round": server_round}


def get_evaluate_fn(model: LinearRegression):
    _, (X_test, y_test) = utils.load_data()

    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        utils.set_model_params(model, parameters)
        loss = mean_squared_error(y_test,model.predict(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.server.start_server(
        server_address="localhost:8000",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=100),
    )
