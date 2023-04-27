import socket
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")

soc = socket.socket()
logging.info("Socket is created")

soc.connect(('localhost', 10000))
logging.info("Connected to the server")

msg = "A message from the client."
msg = pickle.dumps(msg)
soc.sendall(msg)
logging.info("Client sent a message to the server.")

received_data: bytes = b""
while str(received_data)[-2] != ".":
    data = soc.recv(8)
    received_data += data

received_data = pickle.loads(received_data)
logging.info(f"Received data from the client: {received_data}")

soc.close()
print("Socket is closed.")
