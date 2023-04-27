# Guide

Training a `Logistic Regression` model n MNIST using Flower and Scikit-learn.

This code consists of one *server* and two *clients* all having the same model.

- Clients generate individual model parameter updates for the model based on their local datasets.
- These updates are sent to the server which will aggregate them to produce an updated global model.
- Finally, the server sends this improved model back to each client.
- A complete cycle of parameters updates is called a *round*

## 1. Create a virtual environment
```sh
python -m venv venv
```
## 2. Activate the virtual environment
```sh
venv\Scripts\activate
```
## 3. Install Flower and Scikit-Learn
```sh
pip install flwr scikit-learn
```
## 4. Open a command prompt and run the server
```sh
python server.py
```
## 5. Open another 2 command prompts, and run two clients
```sh
python client.py
```
Now, you'll see the process in action.