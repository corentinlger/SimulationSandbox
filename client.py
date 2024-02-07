import socket
import pickle

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.utils.network import SERVER

print(f"{SERVER = }")
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

def receive_loop():
    print(f"entered receive loop")
    while True:
        try:
            raw_data = client.recv(DATA_SIZE)
            data = pickle.loads(raw_data)
            print(f"data received: {data}")
            state, color = data
            print(f"{color = }")
            Simulation.visualize_sim(state, color)

        except socket.error as e:
            print(e)
            client.close()
            break

receive_loop()