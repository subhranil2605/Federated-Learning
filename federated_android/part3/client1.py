import socket
import pickle
import numpy as np

import pygad
import pygad.nn
import pygad.gann


def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(
        last_layer=GANN_instance.population_networks[sol_idx], data_inputs=data_inputs)
    correct_predictions = np.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions / data_outputs.size) * 100

    return solution_fitness


def callback_generation(ga_instance):
    global GANN_instance, last_fitness
    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(
        population_trained_weights=population_matrices)

    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")

    last_fitness = ga_instance.best_solution()[1]


# last fitness
last_fitness = 0


def prepare_GA(GANN_instance):
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=GANN_instance.population_networks)

    initial_population = population_vectors.copy()
    num_parents_mating = 4
    num_generations = 500
    mutation_percent_genes = 5
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    keep_parents = 1
    init_range_low = -2
    init_range_high = 5

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        initial_population=initial_population,
        fitness_func=fitness_func,
        mutation_percent_genes=mutation_percent_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        keep_parents=keep_parents,
        callback_generation=callback_generation
    )

    return ga_instance


def recv(soc, buffer_size=1024, recv_timeout=10):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print(f"A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds. There may be an error or the model may be trained successfully.")
            return None, 0
        except BaseException as e:
            print(f"An error occurred while receiving data from the server {e}.")
            return None, 0

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print(f"Error Decoding the Client's Data: {e}.\n")
        return None, 0

    return received_data, 1



data_inputs = np.array([
    [0, 1],
    [0, 0]
])

data_outputs = np.array([1,0])


HOST = "localhost"
PORT = 10001

soc =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket is created.")


try:
    soc.connect((HOST, PORT))
    print("Connected to the server.")
except BaseException as e:
    print(f"Error Connecting to the Server: {e}")
    soc.close()
    print("Socket Closed.")


subject = "echo"
GANN_instance = None
best_sol_idx = -1


while True:
    data = {"subject": subject, "data": GANN_instance, "best_solution_idx": best_sol_idx}
    data_bytes = pickle.dumps(data)

    print("Sending the Model to the Server.\n")
    soc.sendall(data_bytes)

    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=1024, 
                                 recv_timeout=10) 

    if status == 0:
        print("Nothing Received from the Server.")
        break
    else:
        print(received_data, end="\n\n")
    
    subject = received_data["subject"]
    if subject == "model":
        GANN_instance = received_data["data"]
    elif subject == "done":
        print("The server said the model is trained successfully and no need for further updates its parameters.")
        break
    else:
        print("Unrecognized message type.")
        break

    ga_instance = prepare_GA(GANN_instance)
    ga_instance.run()
    ga_instance.plot_result()

    subject = "model"
    best_sol_idx = ga_instance.best_solution()[2]


soc.close()
print("Socket is closed.")
