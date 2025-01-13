import socket
from _thread import *
import pickle
import argparse
import sys
from shared_data import *
from enum import Enum


class ServerPhase(Enum):
    WAITING_FOR_CONN = 0
    PICKING = 1
    PICKED = 2
    GUESSING = 3
    WAITING = 4


class PlayerState:
    def __init__(self):
        self.phase = ServerPhase.WAITING_FOR_CONN

    def get_phase(self) -> ServerPhase:
        return self.phase

    def set_conn(self, conn):
        self.conn = conn
        self.phase = ServerPhase.PICKING

    def get_conn(self):
        return self.conn

    def set_number(self, number: int):
        self.number = number
        self.phase = ServerPhase.PICKED


class GameState:
    def __init__(self):
        # keep track of two players
        self.player_state = [PlayerState(), PlayerState()]

    def get_player_state(self, index: int) -> PlayerState:
        return self.player_state[index]


def process_NumberPickData(
    current_state: GameState, input: NumberPickData, player: int
) -> GameState | Error:
    player_state = current_state.get_player_state(player)
    if player_state.get_phase() != ServerPhase.PICKING:
        return Error("Got NumberPickData when player was not picking")

    number = input.GetNumber()
    if number < 1 or number > 100:
        return Error("{} is not in the range 1 to 100".format(number))

    player_state.set_number(number)
    player_state.get_conn().send(
        pickle.dumps(Message("Your number is set. Waiting for the other player"))
    )
    return current_state


def process_input(
    current_state: GameState, input: any, player: int
) -> GameState | Error:
    match input:
        case NumberPickData() as data:
            return process_NumberPickData(current_state, data, player)
        case _:
            print("recieved an unknown type")
            return Error("server recieved an unknown type")


def threaded_client(conn, state: GameState, player: int):
    state.get_player_state(player).set_conn(conn)
    initial_state = ClientGameState(
        ClientPhase.PICKING, "Pick a number between 1 and 100"
    )
    conn.send(pickle.dumps(initial_state))
    while True:
        try:
            input = pickle.loads(conn.recv(2048))

            if not input:
                print("Disconnected")
                break

            result = process_input(state, input, player)

            match result:
                case GameState() as new_state:
                    state = new_state
                case Error() as error:
                    conn.sendall(pickle.dumps(error))

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
