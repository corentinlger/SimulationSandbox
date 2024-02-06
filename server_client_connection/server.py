import time
import socket 
import pickle
import threading 

import numpy as np 

SERVER = '10.204.2.189'
# SERVER = '192.168.1.24'
PORT = 5050
ADDR = (SERVER, PORT)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)
server.listen()
print(f"Server started and listening ...")


def generate_random_array(size=(5, 5)):
    return np.random.randint(0, 10, size=size)

def update_latest_data():
    global latest_data
    while True:
        with data_lock:
            latest_data = generate_random_array(array_size)
            new_data_event.set()
        time.sleep(1)

def update_array_size(new_size):
    global array_size
    array_size = new_size

def handle_client(client, addr):
    try:
        client.send("RECEIVE_OR_UPDATE".encode())
        response = client.recv(1024).decode()
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

        elif response == "UPDATE":
            while True:
                try:
                    new_size = pickle.loads(client.recv(1024))
                    update_array_size(new_size)
                except socket.error as e:
                    print(f"error: {e}")
                    client.close()
                    print(f"Client {client} disconnected")

        else:
            print(f"Unknown connection type {response} detected")


    except socket.error as e:
        print(f"error: {e}")
        client.close()
        print(f"Client {client} disconnected")


# Create a global variable to store the current array size and data + lock and event to access it
array_size = (5, 5)
latest_data = generate_random_array(array_size)
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

