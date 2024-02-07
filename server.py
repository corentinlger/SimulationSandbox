import time
import socket 
import pickle
import threading 

import jax

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.agents import Agents
from MultiAgentsSim.utils.network import SERVER


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
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
STEP_DELAY = 0.3
SEED = 0


key = jax.random.PRNGKey(SEED)
sim = Simulation(MAX_AGENTS, GRID_SIZE)
agents = Agents(MAX_AGENTS, GRID_SIZE)

# Create initial simulation state 
grid = sim.init_grid(GRID_SIZE)
agents_pos, agents_states, num_agents = agents.init_agents(NUM_AGENTS, MAX_AGENTS, key)
color = (1.0, 0.0, 0.0)


def get_new_timestep_data(timestep, grid, agents_pos, agents_states, num_agents, color, key):
    key, a_key, add_key = jax.random.split(key, 3)
    actions = agents.choose_action(agents_pos, a_key)
    agents_pos = sim.move_agents(agents_pos, actions)
    agents_states += 0.1
    return (timestep, grid, agents_pos, agents_states, num_agents, color, key)


def update_latest_data():
    global latest_data
    global timestep
    while True:
        with data_lock:
            timestep, grid, agents_pos, agents_states, num_agents, color, key = latest_data
            latest_data = get_new_timestep_data(timestep, grid, agents_pos, agents_states, num_agents, color, key)
            new_data_event.set()
        timestep += 1
        time.sleep(STEP_DELAY)


def update_color(new_color):
    global latest_data
    latest_data = latest_data[:5] + (new_color,) + (latest_data[6],)


def handle_client(client, addr):
    try:
        client.send("RECEIVE_OR_UPDATE".encode())
        response = client.recv(DATA_SIZE).decode()
        print(f"{response} established with {addr}")

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


# Create a global variable to store the current array size and data + lock and event to access it
timestep = 0
latest_data = get_new_timestep_data(timestep, grid, agents_pos, agents_states, num_agents, color, key)
data_lock = threading.Lock()
new_data_event = threading.Event()

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

