import pickle
import socket

servers = ['localhost', '10.204.2.210', '10.204.2.229', '10.204.2.189', '192.168.1.24']
server_idx = 0

SERVER = servers[server_idx]


# Establish a connection with a client
def establish_connection(client, addr, simulation, sim_type, data_size):
    try:
        client.send(sim_type.encode())
        client.send(pickle.dumps(simulation.state))
        connection_type = client.recv(data_size).decode()
        print(f"{connection_type} connection established with {addr}")
        return connection_type

    except socket.error as e:
        print(f"error: {e}")
        client.close()
        print(f"Client {client} disconnected")
