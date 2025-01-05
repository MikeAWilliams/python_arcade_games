import sys
from client_network import Network
from shared_data import *
import argparse


def main(host: str, port: int) -> int:
    coms = Network(host, port)
    initial_state = coms.connect()
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
