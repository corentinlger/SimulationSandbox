import socket 
import pickle
import time 

import matplotlib.pyplot as plt
import numpy as np

SERVER = '10.204.2.189'
PORT = 8080
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)


def plot_image(data_variable, delay=0.1):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))

        plt.clf()

        plt.imshow(data_variable, cmap='viridis')
        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        plt.draw()
        plt.pause(delay)

try:
    while True:
        data = client.recv(4096)
        if not data:
            break

        data_variable = pickle.loads(data)
        print(f"Received {data_variable = }")
        plot_image(data_variable)

except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
