import socket
import pickle
from shared_data import *


def send(connection: socket.socket, data: any):
    connection.send(pickle.dumps(data))


def recieve(connection: socket.socket) -> any:
    return pickle.loads(connection.recv(2048))
