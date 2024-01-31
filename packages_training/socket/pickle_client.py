import socket, pickle
from data import ProcessData

import jax.numpy as jnp

SERVER = '10.204.2.189'
PORT = 8080
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

variable = ProcessData()
variable = jnp.arange(100)
data_string = pickle.dumps(variable)
client.send(data_string)

client.close()
print('Data Sent to Server')
