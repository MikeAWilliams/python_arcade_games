import socket
from _thread import *
import pickle
import argparse
import sys
from shared_data import *


class PlayerState:
    pass


class GameState:
    def __init__(self):
        # keep track of two players
        self.player_state = [PlayerState(), PlayerState()]


def threaded_client(conn, state: GameState, player: int):
    initial_state = InitialState("hello from the server")
    conn.send(pickle.dumps(initial_state))
    while True:
        try:
            data = pickle.loads(conn.recv(2048))

            if not data:
                print("Disconnected")
                break

            # Use this to respond to different kinds of input
            match data:
                case InputType1():
                    conn.sendall(pickle.dumps(ClientGameState("got a 1")))
                case InputType2():
                    conn.sendall(pickle.dumps(ClientGameState("got a 2")))

        except:
            break

    print("Lost connection")
    conn.close()


def main(host: str, port: int, timeout: int) -> int:
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mySocket.settimeout(timeout)  # seconds

    try:
        mySocket.bind((host, port))
    except mySocket.error as e:
        print(str(e))
        return

    mySocket.listen(2)
    print("Waiting for a players to connect")

    game_state = GameState()
    currentPlayer = 0
    while currentPlayer < 1:
        # needs a timeout. blocks forever and ingores ctrl+c
        conn, addr = mySocket.accept()
        print("Connected to:", addr)

        start_new_thread(threaded_client, (conn, game_state, currentPlayer))
        currentPlayer += 1

    # keep the program allive after we got two connections accept ctrl+c
    while True:
        continue

    # unreachable but feels nice to put it here
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", "-a", required=True, type=str, help="host to connect to"
    )
    parser.add_argument(
        "--port", "-p", required=True, type=int, help="port to connect to"
    )
    parser.add_argument(
        "--timeout",
        "-t",
        required=False,
        default="10",
        type=int,
        help="timeout for socket.accept in seconds",
    )
    args = parser.parse_args()
    print(
        "host: {0} port: {1} timeout: {2}".format(args.address, args.port, args.timeout)
    )

    sys.exit(main(args.address, args.port, args.timeout))
