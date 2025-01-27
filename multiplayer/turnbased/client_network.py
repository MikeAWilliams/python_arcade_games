import socket
import pickle
from shared_data import *


class ClientNetwork:
    def __init__(self, host: str, port: int):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (host, port)

    def connect(self):
        try:
            self.client.connect(self.addr)
            return pickle.loads(self.client.recv(2048))
        except Exception as err:
            print(f"Exception in network::connect {err=}, {type(err)=}")
            raise

    # send and recieve an object
    def send_recieve(self, data):
        try:
            self.client.send(pickle.dumps(data))
            return pickle.loads(self.client.recv(2048))
        except socket.error as err:
            print(f"Exception in network::send_recieve {err=}, {type(err)=}")
            raise
        except Exception as err:
            print(f"Exception in network::send_recieve {err=}, {type(err)=}")
            raise

    # send an object, do not recieve
    def send(self, data):
        try:
            self.client.send(pickle.dumps(data))
        except socket.error as err:
            print(f"Exception in network::send {err=}, {type(err)=}")
            raise
        except Exception as err:
            print(f"Exception in network::send {err=}, {type(err)=}")
            raise

    # block until we get something from the server, return it
    def recieve(self):
        return pickle.loads(self.client.recv(2048))
