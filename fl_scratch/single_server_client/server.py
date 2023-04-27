import socket
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")

soc = socket.socket()
logging.info("Socket is created")

soc.bind(('localhost', 10000))
logging.info("Socket is bound to an address and a port number")

# here 1 means, the number of unaccepted connections that the system will allow before refusing new connections
soc.listen(1)
logging.info("Listening for incoming connection...")

connected = False
accept_timeout = 100
soc.settimeout(accept_timeout)  # adding a timeout exception

try:
    conn, addr = soc.accept()
    logging.info(f"Connected to a client: {addr}")
    connected = True
except socket.timeout:
    logging.info(
        f"A socket.timeout exception occurred because the server did not receive any connection for {accept_timeout} seconds"
    )

# data receiving
received_data = b""
if connected:
    while str(received_data)[-2] != ".":
        data = conn.recv(8)
        received_data += data
    received_data = pickle.loads(received_data)
    logging.info(f"Received data from the client: {received_data}")

    msg = "Reply to the server"
    msg = pickle.dumps(msg)
    conn.sendall(msg)
    logging.info("Server sent a message to the client")

    conn.close()
    logging.info(f"Connection is closed with: {addr}")

soc.close()
logging.info("Socket is closed")
