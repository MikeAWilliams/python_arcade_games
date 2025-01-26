import sys
from client_network import ClientNetwork
from shared_data import *
import argparse


def handle_connection_data(data) -> ClientGameState:
    match data:
        case ClientGameState() as state:
            return state
        case Error() as error:
            print("got an error ", error.GetMessage())
            raise Exception(error.GetMessage())
        case _:
            print("recieved an unknown type")
            raise Exception("an unknown error occured on initial connection")


def get_number() -> int:
    while True:
        try:
            string_in = input("Enter your number ")
            number = int(string_in)
            return number
        except ValueError:
            print("{} is not a valid number".format(string_in))


def pick_my_number(coms: ClientNetwork, server_message: str) -> ClientGameState:
    while True:
        print("The server says ", server_message)
        response = coms.send_recieve(NumberPickData(get_number()))
        match response:
            case Error() as error:
                print("error from server {}".format(error.GetMessage()))
            case ClientGameState() as new_state:
                return new_state
            case _:
                raise Exception("recieved an unknown response from the server")


def guess_other_players_number(coms: ClientNetwork, state: ClientGameState):
    print("The server says ", state.GetMessage())
    print("Your number is ", state.GetMyNum())
    print("Their past geusses are ", state.GetTheirGuesses())
    print("Your past geusses are ", state.GetMyGuesses())
    coms.send(GuessData(get_number()))


def main(host: str, port: int) -> int:
    coms = ClientNetwork(host, port)
    initial_state = handle_connection_data(coms.connect())
    if initial_state.GetPhase() != ClientPhase.PICKING:
        raise Exception(
            "Got an unexpected initial state from server {}".format(
                initial_state.GetPhase()
            )
        )

    # picking a number
    post_pic_state = pick_my_number(coms, initial_state.GetMessage())
    print("server says ", post_pic_state.GetMessage())

    while True:
        response = coms.recieve()
        match response:
            case Error() as error:
                print("error from server {}".format(error.GetMessage()))
            case ClientGameState() as new_state:
                match new_state.GetPhase():
                    case ClientPhase.YOU_WIN:
                        print("Yay yay yay! ", new_state.GetMessage())
                        return  # hard stop
                    case ClientPhase.YOU_LOOSE:
                        print("Bummer ", new_state.GetMessage())
                        return  # hard stop
                    case ClientPhase.GUESSING:
                        guess_other_players_number(coms, new_state)
                    case ClientPhase.WAITING_FOR_SERVER:
                        print("server says ", new_state.GetMessage())
                    case _:
                        raise Exception(
                            "got an unexpected phase from the server ",
                            new_state.GetPhase(),
                        )
            case _:
                raise Exception("recieved an unknown response from the server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", "-a", required=True, type=str, help="host to connect to"
    )
    parser.add_argument(
        "--port", "-p", required=True, type=int, help="port to connect to"
    )
    args = parser.parse_args()
    print("host: {0} port: {1}".format(args.address, args.port))

    sys.exit(main(args.address, args.port))
