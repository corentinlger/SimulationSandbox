import socket
import pickle

SERVER = '10.204.2.189'
PORT = 5051

update_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
update_client.connect((SERVER, PORT))
print("Connected to server")


while True:
    try:
        size = input("Enter a size: ")
        tuple_size = tuple(int(size) for size in size.split(" "))
        update_client.send(pickle.dumps(tuple_size))
    except socket.error as e:
        print(f"error: {e}")
        update_client.close()
        break