import sys
from client_network import Network
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


def main(host: str, port: int) -> int:
    coms = Network(host, port)
    initial_state = handle_connection_data(coms.connect())
    if initial_state.GetPhase() != ClientPhase.PICKING:
        raise Exception(
            "Got an unexpected initial state from server {}".format(
                initial_state.GetPhase()
            )
        )

    # picking a number
    while True:
        print("The server says ", initial_state.GetMessage())
        response = coms.send(NumberPickData(get_number()))
        match response:
            case Error() as error:
                print("error from server {}".format(error.GetMessage()))
            case Message() as message:
                print("server says ", message.GetMessage())
                break
            case _:
                raise Exception("recieved an unknown response from the server")

    while True:
        response = coms.send(InputType2())
        print("response to 2 ", response.GetMessage())


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
