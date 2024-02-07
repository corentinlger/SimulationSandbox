import socket
import pickle

from MultiAgentsSim.utils.network import SERVER


PORT = 5050
DATA_SIZE = 4096

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
            color = input("Enter a color in rgb format: ")
            tuple_color = tuple(float(value) for value in color.split(" "))
            print(f"{tuple_color = }")
            update_client.send(pickle.dumps(tuple_color))
            print(f"sent {tuple_color} to server")
        except socket.error as e:
            print(f"error: {e}")
            update_client.close()
            break

update_color_loop()