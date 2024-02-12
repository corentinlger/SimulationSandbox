import time
import socket 
import pickle
import threading 
import argparse

from jax import random
from flax import serialization

from MultiAgentsSim.simple_simulation import SimpleSimulation
from MultiAgentsSim.utils.network import SERVER

parser = argparse.ArgumentParser()
parser.add_argument('--step_delay', type=float, default=0.5)
args = parser.parse_args()


# Initialize server parameters
PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 10835 # size of data that is being transfered at each timestep

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


# Initialize simulation parameters
NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
VIZUALIZE = True
STEP_DELAY = args.step_delay
print(f"{STEP_DELAY = }")

SEED = 0
key = random.PRNGKey(SEED)
color = (1.0, 0.0, 0.0)

sim = SimpleSimulation(MAX_AGENTS, GRID_SIZE)
state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

# Create a global variable to store the current array size and data + lock and event to access it
latest_data = state
data_lock = threading.Lock()
new_data_event = threading.Event()

# Create a function to continuously update the state of the simulation
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

# def update_color(new_color):
#     global latest_data
#     latest_data[1] = new_color

def handle_client(client, addr):
    try:
        client.send("RECEIVE_OR_UPDATE".encode())
        client.send(pickle.dumps(latest_data))
        print(f"{len(pickle.dumps(latest_data)) = }")
        response = client.recv(DATA_SIZE).decode()
        print(f"{response} connection established with {addr}")

        if response == "RECEIVE":
            while True:
                try:
                    new_data_event.wait()
                    with data_lock:
                        sent = serialization.to_bytes(latest_data)
                        client.send(sent)   
                        # print(f"{len(sent) = }")
                        test = serialization.from_bytes(latest_data, sent)
                        # print(f"{test = }")
                    new_data_event.clear()
                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {client} disconnected")
                    break

        # elif response == "UPDATE":
        #     while True:
        #         try:
        #             new_color = pickle.loads(client.recv(DATA_SIZE))
        #             print(f"received new color {new_color} from client")
        #             update_color(new_color)
        #         except socket.error as e:
        #             print(f"error: {e}")
        #             client.close()
        #             print(f"Client {client} disconnected")
        #             break

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