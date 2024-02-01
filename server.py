import socket 
import pickle 
import threading 
import time 

import numpy as np 

def generate_random_array():
    return np.random.random((10, 10))

def handle_client(conn, addr):
    print(f"Connected by {addr}")

    for i in range(10):
        data_variable = generate_random_array()
        pickled_data = pickle.dumps(data_variable)
        conn.send(pickled_data)
        print(f"Sent {pickled_data = }")
        time.sleep(1)

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

        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
        server.close()

except Exception as e:
    print(f"Error: {e}")

finally:
    server.close()
