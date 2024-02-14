import time
import socket 
import pickle
import threading 
import argparse

from jax import random
from flax import serialization

from simulationsandbox.wrapper import SimulationWrapper
from simulationsandbox.utils.network import SERVER
from simulationsandbox.sim_types import SIMULATIONS

parser = argparse.ArgumentParser()
parser.add_argument('--sim_type', type=str, default="two_d")
parser.add_argument('--step_delay', type=float, default=0.1)
args = parser.parse_args()


# Networking constants
PORT = 5050
ADDR = (SERVER, PORT)
DATA_SIZE = 40000
STEP_DELAY = args.step_delay
# Simulation constants
SIM_TYPE = args.sim_type
SEED = 0
NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 

# Initialize server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


# Initialize simulation
Simulation = SIMULATIONS[SIM_TYPE]
key = random.PRNGKey(SEED)
sim = Simulation(MAX_AGENTS, GRID_SIZE)
state = sim.init_state(NUM_AGENTS, NUM_OBS, key)
state_byte_size = len(serialization.to_bytes(state))


# Shared variables and locks
sim_lock = threading.Lock()
update_event = threading.Event()


simulation = SimulationWrapper(sim, state, key, step_delay=STEP_DELAY, update_event=update_event)
print(f"{len(pickle.dumps(simulation.state))}")


# Establish a connection with a client
def establish_connection(client, addr):
    try:
        client.send(SIM_TYPE.encode())
        client.send(pickle.dumps(simulation.state))
        connection_type = client.recv(DATA_SIZE).decode()
        print(f"{connection_type} connection established with {addr}")
        return connection_type

    except socket.error as e:
        print(f"error: {e}")
        client.close()
        print(f"Client {client} disconnected")


# Define how to communicate with a client
def communicate_with_client(client, addr, connection_type):
    global state

    if connection_type == "RECEIVE":
            while True:
                try:
                    update_event.wait()
                    # with data_lock:
                    client.send(serialization.to_bytes(simulation.state))   
                    update_event.clear()

                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {addr} disconnected")
                    break

    elif connection_type == "NOTEBOOK":
            while True:
                try:
                    request = client.recv(DATA_SIZE).decode()
                    print(f"{request}")
                    if request == "CLOSE_CONNECTION":
                        client.close()
                        print(f"Client {addr} disconnected")

                    elif request == "GET_STATE":
                        with sim_lock:
                            client.send(serialization.to_bytes(simulation.state))   

                    elif request == "SET_STATE":
                        with sim_lock:
                            client.send(serialization.to_bytes(simulation.state))  
                            updated_state = serialization.from_bytes(state, client.recv(state_byte_size))  
                            simulation.state = updated_state
                    
                    else:
                        print(f"Unknow request type {request}")

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
    communicate_with_client(client, addr, connection_type)


# Start the simulation
simulation.start()
print("Simulation started")

# Start listening to clients and launch their threads 
while True:
    try:
        client, addr = server.accept()
        print(f"Connected with {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client, addr))
        client_thread.start()

    except socket.error as e:
        print(f"error: {e}")