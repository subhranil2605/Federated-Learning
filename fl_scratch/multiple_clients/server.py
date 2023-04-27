import socket
import pickle
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")


class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, num_rounds, buffer_size=1024, recv_timeout=100):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.num_rounds = num_rounds
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data: bytes = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b"":
                    received_data = b""
                    if time.perf_counter() - self.recv_start_time > self.recv_timeout:
                        return None, 0

                elif str(data)[-2] == ".":
                    logging.info(f"All data ({len(received_data)} bytes) Received from {self.client_info}")

                    if len(received_data) > 0:
                        try:
                            received_data = pickle.loads(received_data)
                            return received_data, 1
                        except BaseException as e:
                            logging.info(f"Error decoding the Client's data: {e}")
                            return None, 0

                else:
                    self.recv_start_time = time.perf_counter()

            except BaseException as e:
                logging.info(f"Error decoding the Client's data: {e}")
                return None, 0

    def run(self):
        while True:
            self.recv_start_time = time.perf_counter()
            logging.info(f"Waiting to Receive Data")

            msg: str = "Reply from the server"
            msg: bytes = pickle.dumps(msg)
            self.connection.sendall(msg)
            logging.info("Server sent a message to the client.")

            received_data, status = self.recv()

            if status == 0:
                self.connection.close()
                logging.info(
                    f"Connection closed with {self.client_info} either due to inactivity for {self.recv_timeout} seconds or due to an error."
                )
                break

            msg: str = "Reply from the server"
            msg: bytes = pickle.dumps(msg)
            self.connection.sendall(msg)
            logging.info("Server sent a message to the client.")


if __name__ == '__main__':
    soc = socket.socket()
    logging.info("Socket is created")

    soc.bind(("localhost", 10000))
    logging.info("Socket is bound to an address and port")

    soc.listen(1)
    logging.info("Listening for incoming connection...")

    while True:
        try:
            conn, addr = soc.accept()
            logging.info(f"New connection from {addr}")

            socket_thread = SocketThread(
                connection=conn,
                client_info=addr,
                buffer_size=1024,
                recv_timeout=100
            )
            socket_thread.start()
        except Exception:
            soc.close()
            logging.info("(Timeout) Socket closed because no connections received.")
            break
