import socket 
import pickle

HEADER = 64 
PORT = 8080
FORMAT = 'utf-8'
SERVER = '10.204.2.189'
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)

send("Hello World")
send("Hello Swag")
send("Hello Yo")
send("!DISCONNECT")