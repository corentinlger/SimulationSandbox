import time
import socket
import pickle

from flax import serialization

from MultiAgentsSim.two_d_simulation import SimpleSimulation
from MultiAgentsSim.utils.network import SERVER

PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 10835
EVAL_TIME = 10

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
print(f"Connected to {ADDR}")

msg = client.recv(1024).decode()
state_example = pickle.loads(client.recv(DATA_SIZE))
state_bytes_size = len(serialization.to_bytes(state_example))
response = "RECEIVE"
client.send(response.encode())
time.sleep(1)


def receive_loop():
    i = 0 
    while True:
        try:
            i += 1 
            raw_data = client.recv(state_bytes_size)
            state = serialization.from_bytes(state_example, raw_data)
            SimpleSimulation.visualize_sim(state, grid_size=None)
        
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
        SimpleSimulation.visualize_sim(state, grid_size=None)
    client.close()

    print(f"{i = } : {i / EVAL_TIME } data received per second")

# test()
receive_loop()