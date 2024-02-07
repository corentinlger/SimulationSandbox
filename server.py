import time
import socket 
import pickle
import threading 

import jax
from jax import random

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.agents import Agents
from MultiAgentsSim.utils.network import SERVER

# Initialize server parameters
PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 4096 # size of data that is being transfered at each timestep

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


# Initialize simulation parameters
NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
STEP_DELAY = 1
SEED = 0
key = random.PRNGKey(SEED)
color = (1.0, 0.0, 0.0)

sim = Simulation(MAX_AGENTS, GRID_SIZE)
state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

# Create a global variable to store the current array size and data + lock and event to access it
latest_data = [state, color]
data_lock = threading.Lock()
new_data_event = threading.Event()


def update_latest_data():
    global latest_data
    global state
    global key

    while True:
        with data_lock:
            key, a_key, step_key = random.split(key, 3)
            actions = sim.choose_action(state.obs, a_key)
            state = sim.step(state, actions, step_key)
            latest_data[0] = state
            new_data_event.set()
        time.sleep(STEP_DELAY)


def update_color(new_color):
    global latest_data
    latest_data[1] = new_color


def handle_client(client, addr):
    try:
        client.send("RECEIVE_OR_UPDATE".encode())
        response = client.recv(DATA_SIZE).decode()
        print(f"{response} connection established with {addr}")

        if response == "RECEIVE":
            while True:
                try:
                    new_data_event.wait()
                    with data_lock:
                        data = latest_data
                    client.send(pickle.dumps(data))
                    new_data_event.clear()
                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {client} disconnected")
                    break

        elif response == "UPDATE":
            while True:
                try:
                    new_color = pickle.loads(client.recv(DATA_SIZE))
                    print(f"received new color {new_color} from client")
                    update_color(new_color)
                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {client} disconnected")
                    break

        else:
            print(f"Unknown connection type {response} detected")


    except socket.error as e:
        print(f"error: {e}")
        client.close()
        print(f"Client {client} disconnected")



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

