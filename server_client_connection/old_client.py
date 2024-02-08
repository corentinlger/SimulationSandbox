import socket
import pickle
import threading

SERVER = '10.204.2.189'
# SERVER = '192.168.1.24'
PORT = 5050
ADDR = (SERVER, PORT)

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
            raw_data = client.recv(1024)
            data = pickle.loads(raw_data)
            print(f"data received: {data}")

        except socket.error as e:
            print(e)
            client.close()
            break

receive_thread = threading.Thread(target=receive)
receive_thread.start()