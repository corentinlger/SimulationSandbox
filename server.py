import socket 
import pickle 
import threading 
import time 

from jax import random 

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.agents import Agents

NUM_AGENTS = 5 
MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
VIZ_DELAY = 0.000001
SEED = 0

key = random.PRNGKey(SEED)
sim = Simulation(MAX_AGENTS, GRID_SIZE)
agents = Agents(MAX_AGENTS, GRID_SIZE)

grid = sim.init_grid(GRID_SIZE)
agents_pos, agents_states, num_agents = agents.init_agents(NUM_AGENTS, MAX_AGENTS, key)

def handle_client(conn, addr, agents_pos, num_agents, key):
    print(f"Connected by {addr}")

    for step in range(NUM_STEPS):
        key, a_key, add_key = random.split(key, 3)

        actions = agents.choose_action(agents_pos, a_key)
        agents_pos = sim.move_agents(agents_pos, actions)
        pickled_data = pickle.dumps((grid, agents_pos, num_agents))
        conn.send(pickled_data)
        print(f"Sent data")
        time.sleep(0.5)

    conn.close()

try:
    print(f"Server is listening ...")

    SERVER = '10.204.2.189'
    PORT = 8080
    ADDR = (SERVER, PORT)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen(1)

    while True:
        conn, addr = server.accept()

        client_thread = threading.Thread(target=handle_client, args=(conn, addr, agents_pos, num_agents, key))
        client_thread.start()
        server.close()

except Exception as e:
    print(f"Error: {e}")

finally:
    server.close()
