import socket
import pickle


HOST = "localhost"
PORT = 10001

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    print("Socket is created.")

    soc.connect((HOST, PORT))
    print("Connected to the server.")

    # sending a message
    msg = "A message from the client."
    msg = pickle.dumps(msg)
    soc.sendall(msg)
    print("Client sent a message to the server.")

    # receivin a message
    received_data = b''
    while str(received_data)[-2] != '.':
        data = soc.recv(8)
        received_data += data
    received_data = pickle.loads(received_data)
    print(f"Received data from the client: {received_data}")

    # another message from the client
    msg = "Another message from the client."
    msg = pickle.dumps(msg)
    soc.sendall(msg)
    print("Client sent a message to the server.")

    # receivin a message
    received_data = b''
    while str(received_data)[-2] != '.':
        data = soc.recv(8)
        received_data += data
    received_data = pickle.loads(received_data)
    print(f"Received data from the client: {received_data}")


print("Socket is closed.")
