import socket
import pickle

HOST = "localhost"
PORT = 65432
BUFFER_SIZE = 8


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    print("Socket created.")
    
    soc.connect((HOST, PORT))

    # sending message to the server
    msg = "A message from the client."
    msg = pickle.dumps(msg)
    soc.sendall(msg)
    print("Client sent a message to the server")

    # receiving data from the server
    received_data = b''
    while str(received_data)[-2] != '.':
        data = soc.recv(BUFFER_SIZE)
        received_data += data
    received_data = pickle.loads(received_data)
    print(f"Received Data from the server: {received_data}")
