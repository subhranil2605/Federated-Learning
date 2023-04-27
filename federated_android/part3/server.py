import time
import pickle
import socket
import threading

import pygad
import pygad.nn
import pygad.gann
import numpy as np


model = None

# Preparing the numpy array for the inputs.
data_inputs = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
])

# Preparing the numpy array for the outputs.
data_outputs = np.array([0, 1, 1, 0])


num_classes = 2
num_inputs = 2
num_solutions = 6

GANN_instance = pygad.gann.GANN(
    num_solutions=num_solutions,
    num_neurons_input=num_inputs,
    num_neurons_hidden_layers=[2],
    num_neurons_output=num_classes,
    hidden_activations=["relu"],
    output_activation="softmax"
)

class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5) -> None:
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

                if data == b"":
                    received_data = b""
                    if time.time() - self.recv_start_time > self.recv_timeout:
                        return None, 0

                elif str(data)[-2] == ".":
                    len_ = len(received_data)
                    print(f"All data ({len_} bytes) received from {self.client_info}")

                    if len_ > 0:
                        try:
                            received_data = pickle.loads(received_data)
                            return received_data, 1

                        except BaseException as e:
                            print(f"Error: {e}")
                            return None, 0

                else:
                    self.recv_start_time = time.time()

            except BaseException as e:
                print(f"Error: {e}")
                return None, 0
    
    def model_averaging(self, model, other_model):
        model_weights = pygad.nn.layers_weights(last_layer=model, initial=False)
        other_model_weights = pygad.nn.layers_weights(last_layer=other_model, initial=False)

        new_weights = np.array(model_weights + other_model_weights) / 2

        pygad.nn.update_layers_trained_weights(last_layer=model, final_weights=new_weights)

    def reply(self, received_data):
        global GANN_instance, data_inputs, data_outputs, model
        if isinstance(received_data, dict):
            if "data" in received_data.keys() and "subject" in received_data.keys():
                subject = received_data["subject"]
                print(f"Client's Message Subject is {subject}.")

                print("Replying to the client")
                if subject == "echo":
                    if not model:
                        data = {"subject": "model", "data": GANN_instance}
                    else:
                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        error = np.sum(np.abs(predictions - data_outputs))

                        if error == 0:
                            data = {"subject": "done", "data": None}
                            print("The client asked for the model but it was already trained successfully. There is no need to send the model to the client for retraining.")
                        else:
                            data = {"subject": "done", "data": GANN_instance}
                    
                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print(f"Error Encoding the Message: {e}.\n")
                elif subject == "model":
                    try:
                        GANN_instance = received_data["data"]
                        best_model_idx = received_data["best_solution_idx"]

                        best_model = GANN_instance.population_networks[best_model_idx]
                        if not model:
                            model = best_model
                        else:
                            predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                            error = np.sum(np.abs(predictions - data_outputs))

                            if error == 0:
                                data = {"subject": "done", "data": None}
                                response = pickle.dumps(data)
                                print("The model is trained successfully and no need to send the model to the client for retraining.")
                                return 
                            
                            self.model_averaging(model, best_model)

                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        print(f"Model Predictions: {predictions}")

                        error = np.sum(np.abs(predictions - data_outputs))
                        print(f"Error = {error}\n")

                        if error != 0:
                            data = {"subject": "model", "data": GANN_instance}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)
                            print("\n*****The Model is Trained Successfully*****\n\n")
                    
                    except BaseException as e:
                        print(f"Error Decoding the Client's Data: {e}.\n") 

                else:
                    response = pickle.dumps("Response from the Server")

                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print(f"Error Sending Data to the Client: {e}.\n")

            else:
                print(f"The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {received_data.keys()}.")
        
        else:
            print(f"A dictionary is expected to be received from the client but {type(received_data)} received.")

    def run(self):
        print(f"Running a Thread for the Connection with {self.client_info}.")

        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting to Receive the data starting from {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec} GMT"
            print(date_time)

            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(f"Connection closed with {self.client_info} either due to inactivity for {self.recv_timeout} seconds or due to an error")
                break
            
            # reply
            self.reply(received_data)


if __name__ == "__main__":
    LOCAL_HOST = "localhost"
    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 10001

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket is created.")

    soc.bind((HOST, PORT))
    print(f"Socket is bound to an address: {HOST} and a port number: {PORT}")

    soc.listen()
    print("Listening for incoming connection...")

    # all_data = b""
    while True:
        try:
            connection, client_info = soc.accept()
            print(f"New connection from {client_info}.")

            socket_thread = SocketThread(connection, client_info, buffer_size=1024, recv_timeout=10)
            socket_thread.start()
        except:
            soc.close()
            print("(Timeout) Socket Closed Because no Connections Received.\n")
            break
