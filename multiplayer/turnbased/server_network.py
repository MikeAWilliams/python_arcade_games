import socket
import pickle
from typing import Any
from shared_data import *


def send(connection: socket.socket, data: Any):
    connection.send(pickle.dumps(data))


def recieve(connection: socket.socket) -> Any:
    return pickle.loads(connection.recv(2048))
