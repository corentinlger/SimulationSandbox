import logging
import argparse

from simulationsandbox.utils.network import SERVER_IP, Server

logging.basicConfig(level=logging.INFO)

PORT = 5050
DATA_SIZE = 40000
MAX_AGENTS = 20
SEED = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="aquarium")
    parser.add_argument('--step_delay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_data', action="store_true")
    args = parser.parse_args()

    server = Server(
        server_ip=SERVER_IP,
        port=PORT,
        env_name=args.env_name,
        max_agents=MAX_AGENTS,
        seed=SEED,
        step_delay=args.step_delay,
        data_size=40000
    )
    server.start()
    