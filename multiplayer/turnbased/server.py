import socket
from _thread import *
import pickle
import argparse
import sys


def threaded_client(conn, player: int):
    # conn.send(pickle.dumps(players[player]))
    conn.send(str.encode("hello from the server"))
    reply = ""
    while True:
        try:
            # data = pickle.loads(conn.recv(2048))
            data = conn.recv(2048)

            if not data:
                print("Disconnected")
                break

            # conn.sendall(pickle.dumps(reply))
            print("data recieved ", data.decode())
            print("sending pong")
            conn.sendall(str.encode("pong"))
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

    currentPlayer = 0
    while currentPlayer < 2:
        # needs a timeout. blocks forever and ingores ctrl+c
        conn, addr = mySocket.accept()
        print("Connected to:", addr)

        start_new_thread(threaded_client, (conn, currentPlayer))
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
