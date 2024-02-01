import socket 
import pickle
import time 

from MultiAgentsSim.simulation import Simulation

SERVER = '10.204.2.189'
PORT = 8080
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

try:
    while True:
        data = client.recv(4096)
        if not data:
            break

        grid, agents_pos, num_agents = pickle.loads(data)
        print(f"Received data")
        Simulation.visualize_sim(grid, agents_pos, num_agents)

except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
