import socket
import pickle
import numpy
import threading

import pygad
import pygad.nn
import pygad.gann



def create_socket():
    soc = socket.socket(
        family=socket.AF_INET, type=socket.SOCK_STREAM)
    return soc


def connect(soc, host, port):
    try:
        soc.connect((host, port))

    except BaseException as e:
        print(f"Error Connecting to the Server: {e}")


def recv_train_model(soc):
    global GANN_instance
    recvThread = RecvThread(soc, buffer_size=1024, recv_timeout=10)
    recvThread.start()


def close_socket(soc):
    soc.close()




def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness


def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(
        population_trained_weights=population_matrices)

#    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))

#last_fitness = 0


def prepare_GA(GANN_instance):
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=GANN_instance.population_networks)

    initial_population = population_vectors.copy()

    num_parents_mating = 4
    num_generations = 500  # Number of generations.
    mutation_percent_genes = 5
    parent_selection_type = "sss"  # Type of parent selection.
    crossover_type = "single_point"  # Type of the crossover operator.
    mutation_type = "random"  # Type of the mutation operator.

    keep_parents = 1
    init_range_low = -2
    init_range_high = 5

    ga_instance = pygad.GA(num_generations=num_generations,
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
                           on_generation=callback_generation)

    return ga_instance


# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([1,
                            0])


class RecvThread(threading.Thread):

    def __init__(self, soc, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.soc = soc
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:  # str(received_data)[-2] != '.':
            try:
                self.soc.settimeout(self.recv_timeout)
                received_data += self.soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print(f"A socket.timeout exception occurred because the server did not send any data for {self.recv_timeout} seconds.")
                return None, 0
            except BaseException as e:
                print(f"Error While Receiving Data from the Server: {e}.")
                return None, 0

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            return None, 0

        return received_data, 1

    def run(self):
        global GANN_instance

        subject = "echo"
        GANN_instance = None
        best_sol_idx = -1

        while True:
            data = {"subject": subject, "data": GANN_instance,
                    "best_solution_idx": best_sol_idx}
            data_byte = pickle.dumps(data)

            try:
                self.soc.sendall(data_byte)
            except BaseException as e:
                break

            received_data, status = self.recv()
            if status == 0:
                print("Nothing Received from the Server")
                break
            else:
                print("New Message from the Server")

            subject = received_data["subject"]
            if subject == "model":
                GANN_instance = received_data["data"]
            elif subject == "done":
                print("Model is Trained")
                break
            else:
                print("Unrecognized Message Type: {subject}".format(
                    subject=subject))
                break

            ga_instance = prepare_GA(GANN_instance)

            ga_instance.run()

            subject = "model"
            best_sol_idx = ga_instance.best_solution()[2]


# create
sock = create_socket()

# connect
connect(sock, "localhost", 10000)

recv_train_model(sock)

# close_socket(sock)