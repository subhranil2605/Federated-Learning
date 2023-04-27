import socket
import pickle

HOST = socket.gethostbyname(socket.gethostname())
# HOST = "localhost"
PORT = 65432
BUFFER_SIZE = 8


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    print("Socket created.")
    
    soc.bind((HOST, PORT))
    soc.listen()
    print(f"Listening on {HOST}:{PORT}")
    
    conn, addr = soc.accept()
    with conn:

        # receiving data from the client
        received_data = b''
        while str(received_data)[-2] != '.':
            data = conn.recv(BUFFER_SIZE)
            received_data += data
        received_data = pickle.loads(received_data)
        print(f"Received Data from the client: {received_data}")

        # sending data to the client
        msg = "Reply from the server"
        msg = pickle.dumps(msg)
        conn.sendall(msg)
        print("Message sent to the client")

print("Socket Closed")
