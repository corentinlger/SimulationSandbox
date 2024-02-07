import socket
import pickle
import threading

from MultiAgentsSim.simulation import Simulation

SERVER = '10.204.2.189'
SERVER = '192.168.1.24'

PORT = 5050
ADDR = (SERVER, PORT)

DATA_SIZE = 4096


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
print(f"Connected to {ADDR}")

msg = client.recv(1024).decode()
print(f"server message: {msg}")
response = "RECEIVE"
client.send(response.encode())
print(f"responded: {response}")

def receive():
    while True:
        try:
            raw_data = client.recv(DATA_SIZE)
            data = pickle.loads(raw_data)
            print(f"data received: {data}")
            timestep, grid, agents_pos, agents_states, num_agents, color, key = data
            print(f"{color = }")
            Simulation.visualize_sim(grid, agents_pos, num_agents, color)

        except socket.error as e:
            print(e)
            client.close()
            break

# Matplotlib intreactive doesn't work outside the main thread
# receive_thread = threading.Thread(target=receive)
# receive_thread.start()
        
receive()