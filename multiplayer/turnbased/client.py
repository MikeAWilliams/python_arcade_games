import sys
from client_network import Network
from shared_data import *
import argparse


def handle_connection_data(data) -> InitialState:
    match data:
        case InitialState() as state:
            return state
        case Error() as error:
            print("got an error ", error.GetMessage())
            raise Exception(error.GetMessage())
        case _:
            print("recieved an unknown type")
            raise Exception("an unknown error occured on initial connection")


def main(host: str, port: int) -> int:
    coms = Network(host, port)
    initial_state = handle_connection_data(coms.connect())
    print("initial state message ", initial_state.GetMessage())

    while True:
        response = coms.send(InputType1())
        print("response to 1 ", response.GetMessage())
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
