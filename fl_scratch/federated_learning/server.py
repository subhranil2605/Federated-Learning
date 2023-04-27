import socket
import pickle
import threading
import time
import logging

from . import fashion_mnist
import numpy

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")

model = fashion_mnist.load_model()

# load evaluation data
_, xy_test = fashion_mnist.load_data(partition=0, num_partitions=1)

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([0,
                            1,
                            1,
                            0])

num_classes = 2
num_inputs = 2

num_solutions = 6


# GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
#                                 num_neurons_input=num_inputs,
#                                 num_neurons_hidden_layers=[2],
#                                 num_neurons_output=num_classes,
#                                 hidden_activations=["relu"],
#                                 output_activation="softmax")


class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'':  # Nothing received from the client.
                    received_data: bytes = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0  # 0 means the connection is no longer active and it should be closed.

                elif str(data)[-2] == '.':
                    logging.info(f"All data ({len(received_data)} bytes) Received from {self.client_info}.")

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            logging.warning("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                logging.warning("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, model, other_model):
        model_weights = pygad.nn.layers_weights(last_layer=model, initial=False)
        other_model_weights = pygad.nn.layers_weights(last_layer=other_model, initial=False)
        new_weights = numpy.array(model_weights + other_model_weights) / 2
        pygad.nn.update_layers_trained_weights(last_layer=model, final_weights=new_weights)

    def reply(self, received_data):
        global GANN_instance, data_inputs, data_outputs, model
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                print("Client's Message Subject is {subject}.".format(subject=subject))

                print("Replying to the Client.")
                if subject == "echo":
                    if model is None:
                        data = {"subject": "model", "data": GANN_instance}
                    else:
                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        if error == 0:
                            data = {"subject": "done", "data": None}
                            print(
                                "The client asked for the model but it was already trained successfully. There is no need to send the model to the client for retraining.")
                        else:
                            data = {"subject": "model", "data": GANN_instance}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        GANN_instance = received_data["data"]
                        best_model_idx = received_data["best_solution_idx"]

                        best_model = GANN_instance.population_networks[best_model_idx]
                        if model is None:
                            model = best_model
                        else:
                            predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)

                            error = numpy.sum(numpy.abs(predictions - data_outputs))

                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                            if error == 0:
                                data = {"subject": "done", "data": None}
                                response = pickle.dumps(data)
                                logging.info(
                                    "The model is trained successfully and no need to send the model to the client for retraining.")
                                return

                            self.model_averaging(model, best_model)

                        # print(best_model.trained_weights)
                        # print(model.trained_weights)

                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        print("Model Predictions: {predictions}".format(predictions=predictions))

                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        print("Error = {error}\n".format(error=error))

                        if error != 0:
                            data = {"subject": "model", "data": GANN_instance}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)
                            print("\n*****The Model is Trained Successfully*****\n\n")

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")

                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print(
                    "The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(
                        d_keys=received_data.keys()))
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(
                d_type=type(received_data)))

    def run(self):
        print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            logging.info(f"\nWaiting to Receive Data from {self.client_info}")
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                logging.info(
                    f"\nConnection Closed with {self.client_info} either due to inactivity for {self.recv_timeout} seconds or due to an error.\n\n")
                break

            # print(received_data)
            self.reply(received_data)


soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
logging.info("Socket Created.\n")

# Timeout after which the socket will be closed.
# soc.settimeout(5)

soc.bind(("localhost", 10000))
logging.info("Socket Bound to IPv4 Address & Port Number.\n")

soc.listen(1)
logging.info("Socket is Listening for Connections ....\n")

all_data: bytes = b""
while True:
    try:
        connection, client_info = soc.accept()
        logging.info(f"\nNew Connection from {client_info}.")
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info,
                                     buffer_size=1024,
                                     recv_timeout=10)
        socket_thread.start()
    except:
        soc.close()
        logging.warning("(Timeout) Socket Closed Because no Connections Received.\n")
        break
