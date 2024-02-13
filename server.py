import time
import socket 
import pickle
import threading 
import argparse

from jax import random
from flax import serialization

from MultiAgentsSim.two_d_simulation import SimpleSimulation
from MultiAgentsSim.three_d_simulation import ThreeDSimulation
from MultiAgentsSim.utils.network import SERVER

parser = argparse.ArgumentParser()
parser.add_argument('--step_delay', type=float, default=0.1)
args = parser.parse_args()


# Networking constants
PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 10835
STEP_DELAY = args.step_delay
# Simulation constants
SEED = 0
NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
VIZUALIZE = True

# Initialize server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


# Initialize simulation
key = random.PRNGKey(SEED)
sim = SimpleSimulation(MAX_AGENTS, GRID_SIZE)
state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

# Shared variables and locks
latest_data = state
data_lock = threading.Lock()
new_data_event = threading.Event()


# Continuously update the state of the simulation
def update_latest_data():
    global latest_data
    global state
    global key

    while True:
        key, a_key, step_key = random.split(key, 3)
        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)
        with data_lock:
            latest_data = state
        new_data_event.set()
        time.sleep(STEP_DELAY)


# Establish a connection with a client
def establish_connection(client, addr):
    try:
        client.send("RECEIVE_OR_UPDATE".encode())
        client.send(pickle.dumps(latest_data))
        connection_type = client.recv(DATA_SIZE).decode()
        print(f"{connection_type} connection established with {addr}")
        return connection_type

    except socket.error as e:
        print(f"error: {e}")
        client.close()
        print(f"Client {client} disconnected")


# Define how to communicate with a client
def communicate_with_client(connection_type):
    if connection_type == "RECEIVE":
            while True:
                try:
                    new_data_event.wait()
                    with data_lock:
                        serialized_data = serialization.to_bytes(latest_data)
                        client.send(serialized_data)   
                    new_data_event.clear()

                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {client} disconnected")
                    break

    else:
        print(f"Unknown connection type {connection_type} detected")


# Function to handle a client when it connects to the server 
def handle_client(client, addr):
    connection_type = establish_connection(client, addr)
    communicate_with_client(connection_type)


# Create a thread to continuously update the data
update_data_thread = threading.Thread(target=update_latest_data)
update_data_thread.start()


# Start listening to clients and launch their threads 
while True:
    try:
        client, addr = server.accept()
        print(f"Connected with {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client, addr))
        client_thread.start()

    except socket.error as e:
        print(f"error: {e}")