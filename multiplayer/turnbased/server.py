import socket
from _thread import *
import argparse
import sys
from shared_data import *
from enum import Enum
import random
from typing import List
import server_network


class ServerPhase(Enum):
    WAITING_FOR_CONNECTION = 0
    PICKING = 1
    PICKED = 2
    GUESSING = 3
    WAITING = 4
    WON = 5
    LOST = 6


class PlayerState:
    def __init__(self):
        self.phase = ServerPhase.WAITING_FOR_CONNECTION
        self.past_guesses = []

    def get_phase(self) -> ServerPhase:
        return self.phase

    def set_connection(self, connection: socket.socket):
        self.connection = connection
        self.phase = ServerPhase.PICKING

    def get_connection(self) -> socket.socket:
        return self.connection

    def set_number(self, number: int):
        self.number = number
        self.phase = ServerPhase.PICKED

    def get_number(self):
        return self.number

    def set_your_turn(self):
        self.phase = ServerPhase.GUESSING

    def set_waiting(self):
        self.phase = ServerPhase.WAITING

    def set_won(self):
        self.phase = ServerPhase.WON

    def set_lost(self):
        self.phase = ServerPhase.LOST

    def add_guess(self, guess: int):
        self.past_guesses.append(guess)

    def get_guesses(self) -> List[int]:
        return self.past_guesses


class GameState:
    def __init__(self):
        # keep track of two players
        self.player_state = [PlayerState(), PlayerState()]

    def get_player_state(self, index: int) -> PlayerState:
        return self.player_state[index]

    def get_other_player_state(self, current_index: int) -> PlayerState:
        if current_index == 0:
            return self.player_state[1]
        return self.player_state[0]


class TurnTransitionData:
    def __init__(self):
        self.cp_won = False

    def set_messages(self, cp_message: str, op_message: str):
        self.cp_message = cp_message
        self.op_message = op_message

    def current_player_won(self):
        self.cp_won = True


def change_turn(
    current_player: PlayerState,
    other_player: PlayerState,
    transition: TurnTransitionData,
):
    current_player.set_waiting()
    other_player.set_your_turn()
    server_network.send(
        current_player.get_connection(),
        ClientGameState(
            ClientPhase.WAITING_FOR_SERVER,
            "{}\n{}".format(transition.cp_message, "Its the other players turn."),
            current_player.get_number(),
            current_player.get_guesses(),
            other_player.get_guesses(),
        ),
    )
    server_network.send(
        other_player.get_connection(),
        ClientGameState(
            ClientPhase.GUESSING,
            "{}\n{}".format(transition.op_message, "Its your turn. Guess a number."),
            other_player.get_number(),
            other_player.get_guesses(),
            current_player.get_guesses(),
        ),
    )


def current_player_won(
    current_player: PlayerState,
    other_player: PlayerState,
    transition: TurnTransitionData,
):
    current_player.set_won()
    other_player.set_lost()
    server_network.send(
        current_player.get_connection(),
        ClientGameState(
            ClientPhase.YOU_WIN,
            "{}\n{}".format(transition.cp_message, "Congradulations you win!"),
            current_player.get_number(),
            current_player.get_guesses(),
            other_player.get_guesses(),
        ),
    )
    server_network.send(
        other_player.get_connection(),
        ClientGameState(
            ClientPhase.YOU_LOOSE,
            "{}\n{}".format(transition.op_message, "You loose!"),
            other_player.get_number(),
            other_player.get_guesses(),
            current_player.get_guesses(),
        ),
    )


def change_turn_or_win(
    current_player: PlayerState,
    other_player: PlayerState,
    transition: TurnTransitionData,
):
    if transition.cp_won:
        current_player_won(current_player, other_player, transition)
    else:
        change_turn(current_player, other_player, transition)


def process_GuessData(state: GameState, input: NumberPickData, player: int) -> Error:
    print("processing a guess for player {}".format(player))
    player_state = state.get_player_state(player)
    if player_state.get_phase() != ServerPhase.GUESSING:
        return Error("Got NumberPickData when player was not guessing")

    number = input.GetNumber()
    player_state.add_guess(number)

    other_player_state = state.get_other_player_state(player)
    other_number = other_player_state.get_number()
    transition = TurnTransitionData()
    if number < other_number:
        transition.set_messages(
            "Your guess is to low", "They guessed {} which is to low".format(number)
        )
    elif number > other_number:
        transition.set_messages(
            "Your guess is to high", "They guessed {} which is to high".format(number)
        )
    else:
        print("player {} won".format(player))
        transition.set_messages(
            "You guessed their number",
            "They guessed {} which is your number".format(number),
        )
        transition.current_player_won()

    change_turn_or_win(player_state, other_player_state, transition)
    return None


def notify_player_of_other_picking_status(
    player: int, player_state: PlayerState, other_player_state: PlayerState
) -> bool:
    # there is race condition here. Both players may be in this block of code at the same time
    match (other_player_state.get_phase()):
        case ServerPhase.WAITING_FOR_CONNECTION:
            server_network.send(
                player_state.get_connection(),
                ClientGameState(
                    ClientPhase.WAITING_FOR_SERVER,
                    "Your number is set. Waiting for the other player to connect",
                ),
            )
        case ServerPhase.PICKING:
            server_network.send(
                player_state.get_connection(),
                ClientGameState(
                    ClientPhase.WAITING_FOR_SERVER,
                    "Your number is set. Waiting for the other player to pick",
                ),
            )
        case ServerPhase.PICKED:
            # both players have picked
            server_network.send(
                player_state.get_connection(),
                ClientGameState(
                    ClientPhase.WAITING_FOR_SERVER,
                    "Your number is set. Other player picked before you",
                ),
            )
            # return true to tell caller to move both players to the next step
            return True
        case _:
            print("blowing up for player ", player)
            raise Exception(
                "Other player is in unexpected state ".format(
                    other_player_state.get_phase()
                )
            )
    return False


def move_both_players_out_of_waiting_for_pick(state: GameState):
    # pick a random player to go first
    turn_index = random.choice([0, 1])
    print("picked player {} to go first".format(turn_index))
    turn_player = state.get_player_state(turn_index)
    other_player = state.get_other_player_state(turn_index)
    if (
        turn_player.get_phase() != ServerPhase.PICKED
        and other_player.get_phase() != ServerPhase.PICKED
    ):
        raise Exception(
            "Unexpected state when picking a first player turn_index: {} turn_player.phase: {}, ohter_player.phase {}".format(
                turn_index, turn_player.get_phase(), other_player.get_phase()
            )
        )
    turn_player.set_your_turn()
    other_player.set_waiting()

    server_network.send(
        turn_player.get_connection(),
        ClientGameState(
            ClientPhase.GUESSING,
            "Its your turn. Guess a number.",
            turn_player.get_number(),
        ),
    )
    server_network.send(
        other_player.get_connection(),
        ClientGameState(ClientPhase.WAITING_FOR_SERVER, "Its the other players turn."),
    )


def process_NumberPickData(
    state: GameState, input: NumberPickData, player: int
) -> Error:
    player_state = state.get_player_state(player)
    if player_state.get_phase() != ServerPhase.PICKING:
        raise Exception("Got NumberPickData when player was not picking")

    number = input.GetNumber()
    if number < 1 or number > 100:
        return Error("{} is not in the range 1 to 100".format(number))

    player_state.set_number(number)
    other_player_state = state.get_other_player_state(player)
    both_picked = notify_player_of_other_picking_status(
        player, player_state, other_player_state
    )
    if both_picked:
        move_both_players_out_of_waiting_for_pick(state)

    return None


def process_input(state: GameState, input: any, player: int) -> Error:
    match input:
        case NumberPickData() as data:
            return process_NumberPickData(state, data, player)
        case GuessData() as data:
            return process_GuessData(state, data, player)
        case _:
            print("recieved an unknown type")
            return Error("server recieved an unknown type")


def server_game_loop(connection, state: GameState, player: int):
    state.get_player_state(player).set_connection(connection)
    server_network.send(
        connection,
        ClientGameState(ClientPhase.PICKING, "Pick a number between 1 and 100"),
    )
    while True:
        try:
            input = server_network.recieve(connection)

            if not input:
                print("Disconnected")
                break

            error = process_input(state, input, player)

            if error is not None:
                server_network.send(connection, error)

        except:
            break

    print("Lost connection")
    connection.close()


def main(host: str, port: int, timeout: int) -> int:
    main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    main_socket.settimeout(timeout)  # seconds

    try:
        main_socket.bind((host, port))
    except main_socket.error as e:
        print(str(e))
        return

    main_socket.listen(2)
    print("Waiting for players to connect")

    game_state = GameState()
    currentPlayer = 0
    while currentPlayer < 2:
        # needs a timeout. blocks forever and ingores ctrl+c
        connection, addr = main_socket.accept()
        print("Connected to:", addr)

        start_new_thread(server_game_loop, (connection, game_state, currentPlayer))
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
