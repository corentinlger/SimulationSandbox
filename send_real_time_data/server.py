import time
import socket 
import pickle
import threading 

import numpy as np 

SERVER = '10.204.2.189'
SERVER = '192.168.1.24'
PORT = 5050
ADDR = (SERVER, PORT)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


def generate_random_array():
    return np.random.randint(0, 10, size=(5, 5))

def broadcast(data):
    pickled_data = pickle.dumps(data)
    for client in clients:
        client.send(pickled_data)
    print(f"sent {data} to server")

def handle_client(client):
    while True:
        try:
            data = generate_random_array()
            broadcast(data)
            time.sleep(1)
        except socket.error as e:
            print(f"error: {e}")
            client.close()
            print(f"Client {client} disconnected")
            break

clients = []

while True:
    try:
        client, addr = server.accept()
        clients.append(client)
        print(f"Connected with {addr}")

        client_thread = threading.Thread(target=handle_client, args=(client, ))
        client_thread.start()
    except socket.error as e:
        print(f"error: {e}")

