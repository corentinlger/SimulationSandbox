import socket 
import threading


HEADER = 64
PORT = 8080
SERVER = '10.204.2.189'
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"{ADDR = }")
server.bind(ADDR)

def handle_client(conn, addr):
    print(f"[NEW COONECTION] {addr} connected.")

    connected = True 
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False 

            print(f"[{addr}] {msg}")

    conn.close()
        

def start():
    server.listen()
    print(f"[LISTENING] Server listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


if __name__ == '__main__':
    print("[Starting] server is starting ...")
    start()