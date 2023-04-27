import time
import pickle
import socket
import threading


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

    def run(self):
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting to Receive the data starting from {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec} GMT"
            print(date_time)

            _, status = self.recv()
            if status == 0:
                self.connection.close()
                print(f"Connection closed with {self.client_info} either due to inactivity for {self.recv_timeout} seconds or due to an error")
                break

            msg = "Reply from the server."
            msg = pickle.dumps(msg)
            self.connection.sendall(msg)
            print("Server sent a message to the client.")


if __name__ == "__main__":
    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 10001

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket is created.")

    soc.bind((HOST, PORT))
    print(f"Socket is bound to an address: {HOST} and a port number: {PORT}")

    soc.listen(1)
    print("Listening for incoming connection...")

    while True:
        try:
            connection, client_info = soc.accept()
            print(f"New connection from {client_info}.")

            socket_thread = SocketThread(
                connection, client_info, buffer_size=1024, recv_timeout=10)
            socket_thread.start()
        except:
            soc.close()
            print("(Timeout) Socket Closed Because no Connections Received.\n")
            break
