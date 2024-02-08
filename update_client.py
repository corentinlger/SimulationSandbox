import socket
import pickle

from MultiAgentsSim.utils.network import SERVER


PORT = 5050
DATA_SIZE = 4096
COLORS = {1: "red",
          2: "green",
          3: "blue"
          }

update_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
update_client.connect((SERVER, PORT))
print("Connected to server")

msg = update_client.recv(DATA_SIZE).decode()
print(f"server message: {msg}")
response = "UPDATE"
update_client.send(response.encode())
print(f"responded: {response}")

def update_color_loop():
    while True:
        try:
            color_idx = int(input(f"Enter the color idx of your choice {COLORS}: "))
            print(f"{color_idx = }")
            color = COLORS[color_idx]
            update_client.send(pickle.dumps(color))
            print(f"sent {color} to server")
        except socket.error as e:
            print(f"error: {e}")
            update_client.close()
            break

update_color_loop()