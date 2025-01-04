import socket
import pickle


class Network:
    def __init__(self, host: str, port: int):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (host, port)
        self.connect()

    def connect(self):
        try:
            self.client.connect(self.addr)
            data_from_server = self.client.recv(2048).decode()
            print("connection reply ", data_from_server)
        except Exception as err:
            print(f"Exception in network::connect {err=}, {type(err)=}")
            raise

    # send and recieve a string
    def send_string(self, data: str):
        try:
            # self.client.send(pickle.dumps(data))
            self.client.send(str.encode(data))
            # return pickle.loads(self.client.recv(2048))
            return self.client.recv(2048).decode()
        except socket.error as err:
            print(f"Exception in network::send {err=}, {type(err)=}")
            raise
        except Exception as err:
            print(f"Exception in network::send {err=}, {type(err)=}")
            raise
