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

def update_latest_data():
    global latest_data
    while True:
        with data_lock:
            latest_data = generate_random_array()
            new_data_event.set()
        time.sleep(1)

def handle_client(client):
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

# Create a global variable to store the current data + lock and event to access it
latest_data = generate_random_array()
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

        client_thread = threading.Thread(target=handle_client, args=(client, ))
        client_thread.start()
    except socket.error as e:
        print(f"error: {e}")

