import logging

from simulationsandbox.utils.network import SERVER_IP, ReceiveClient

logging.basicConfig(level=logging.INFO)

PORT = 5050
DATA_SIZE = 40000
EVAL_TIME = 10

if __name__ == "__main__":
    client = ReceiveClient(
        server_ip=SERVER_IP,
        port=PORT,
        data_size=DATA_SIZE,
        eval_time=EVAL_TIME,
        )
    client.connect()
    client.receive_loop()