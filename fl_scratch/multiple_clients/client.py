import socket
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")


def send_mssg(sock: socket, msg: str):
    msg: bytes = pickle.dumps(msg)
    sock.sendall(msg)
    logging.info("Client sent a message to the server.")


def recv_data(sock: socket):
    received_data: bytes = b""
    while str(received_data)[-2] != ".":
        data: bytes = sock.recv(8)
        received_data += data
    received_data = pickle.loads(received_data)
    logging.info(f"Received data from the client: {received_data}")


if __name__ == '__main__':
    soc: socket = socket.socket()
    logging.info("Socket is created.")

    soc.connect(("localhost", 10000))
    logging.info("Connected to the server")

    # receiving generic model
    recv_data(soc)

    send_mssg(soc, "A message from the client")
    recv_data(soc)

    send_mssg(soc, "Another message from the client")
    recv_data(soc)

    soc.close()
    logging.info("Scoket is closed")
