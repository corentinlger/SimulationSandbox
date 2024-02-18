import time
import socket
import pickle

from flax import serialization

from simulationsandbox.utils.network import SERVER
from simulationsandbox.utils.sim_types import SIMULATIONS


PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 40000
EVAL_TIME = 10
GRID_SIZE = 20

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
print(f"Connected to {ADDR}")

sim_type = client.recv(1024).decode()
state_example = pickle.loads(client.recv(DATA_SIZE))
state_bytes_size = len(serialization.to_bytes(state_example))
response = "RECEIVE"
client.send(response.encode())
time.sleep(1)

Simulation = SIMULATIONS[sim_type]


def receive_loop():
    i = 0 
    while True:
        try:
            i += 1 
            raw_data = client.recv(state_bytes_size)
            state = serialization.from_bytes(state_example, raw_data)
            Simulation.visualize_sim(state)
        
        except socket.error as e:
            print(e)
            client.close()
            break


def test():
    start = time.time()
    i = 0 
    while time.time() < start + EVAL_TIME:
        i += 1 
        raw_data = client.recv(state_bytes_size)
        state = serialization.from_bytes(state_example, raw_data)
        Simulation.visualize_sim(state)
    client.close()

    print(f"{i = } : {i / EVAL_TIME } data received per second")

# test()
receive_loop()