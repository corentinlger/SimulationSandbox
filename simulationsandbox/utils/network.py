import time
import socket 
import pickle
import threading 
import logging

from flax import serialization

from simulationsandbox.simulator_wrapper import SimulationWrapper
from simulationsandbox.utils.envs import ENVS

logger = logging.getLogger(__name__)

SERVER_IP = 'localhost'

class Server:
    def __init__(self, server_ip, port, env_name, max_agents, seed, step_delay, data_size):
        self.server_ip = server_ip
        self.port = port 
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.addr = (self.server_ip, self.port)
        self.server.bind(self.addr)
        self.server.listen()

        self.sim_lock = threading.Lock()
        self.update_event = threading.Event()
        self.data_size = data_size
        
        self.env_name = env_name
        self.Env = ENVS[env_name]
        self.env = self.Env(max_agents=max_agents)
        self.state = self.env.init_state(seed=seed)
        self.simulation = SimulationWrapper(self.env, self.state, seed, step_delay=step_delay, update_event=self.update_event)


    def start(self):
        """Start the simulation
        """
        self.simulation.start()
        logger.info("Simulation started")

        # Wait for simulation to start before getting state size in bytes
        time.sleep(1)
        with self.sim_lock:
            self.state_byte_size = len(serialization.to_bytes(self.simulation.state))

        # Start listening to clients and launch their threads
        logger.info("Waiting for clients to connect ...")
        while True:
            try:
                client, addr = self.server.accept()
                logger.info(f"Connected with {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(client, addr))
                client_thread.start()

            except socket.error as e:
                logger.error(f"error: {e}")

    def handle_client(self, client, addr):
        """Get client connection type and start communication with it

        :param client: client
        :param addr: client's adress
        """
        connection_type = self.establish_connection(client, addr)
        self.communicate_with_client(client, addr, connection_type)

    def establish_connection(self, client, client_addr):
        """Establish the connection with a remote client

        :param client: client
        :param client_addr: client's adress
        :return: connection_type of the client (either simple receiver client or notebook)
        """
        try:
            client.send(self.env_name.encode())
            logger.info(f"Sent env name {self.env_name} to client {client_addr}")
            with self.sim_lock:
                client.send(pickle.dumps(self.simulation.state))
                logger.info(f"Sent example state {self.env_name} to client {client_addr}")
            connection_type = client.recv(self.data_size).decode()
            logger.info(f"{connection_type} connection established with {client_addr}")
            return connection_type

        except socket.error as e:
            logger.error(f"error: {e}")
            client.close()
            logger.error(f"Client {client} disconnected")

    def communicate_with_client(self, client, addr, connection_type):
        """Either communicate with a client that only periodically receives new states or with a notebook client that can remotly control the simulator

        :param client: client
        :param connection_type: receive or notebook type
        """
        if connection_type == "RECEIVE":
            self.communicate_receive(client, addr)
        elif connection_type == "NOTEBOOK":
            self.communicate_notebook(client, addr)
        else:
            logger.error(f"Unknown connection type {connection_type} detected")

    def communicate_receive(self, client, addr):
        """Communicate with a client that only receives simulation state periodically

        :param client: client
        :param addr: client's adress
        """
        while True:
            try:
                self.update_event.wait()
                with self.sim_lock:
                    client.send(serialization.to_bytes(self.simulation.state))
                self.update_event.clear()
            except socket.error as e:
                logger.error(f"error: {e}")
                client.close()
                logger.error(f"Client {addr} disconnected")
                break

    def communicate_notebook(self, client, addr):
        """Communicate with a notebook client that can remotly control the simulation

        :param client: notebook client
        :param addr: client's adress
        """
        while True:
            try:
                request = client.recv(self.data_size).decode()
                logger.info(f"notebook client request: {request}")

                if request == "CLOSE_CONNECTION":
                    self.handle_close_connection(client, addr)
                elif request == "GET_STATE":
                    self.handle_get_state(client)
                elif request == "SET_STATE":
                    self.handle_set_state(client)
                elif request == "PAUSE":
                    self.handle_pause()
                elif request == "RESUME":
                    self.handle_resume()
                elif request == "STOP":
                    self.handle_stop()
                elif request == "START":
                    self.handle_start()
                else:
                    logger.error(f"Unknow request type {request}")

            except socket.error as e:
                logger.error(f"error: {e}")
                client.close()
                logger.error(f"Client {client} disconnected")
                break

    # Helper functions for Notebook clients
    def handle_close_connection(self, client, addr):
        """Close connection between the server and the client

        :param client: client
        :param addr: client's adress
        """
        client.close()
        logger.info(f"Client {addr} disconnected")

    def handle_get_state(self, client):
        """Send simulation state to the client

        :param client: client
        """
        with self.sim_lock:
            client.send(serialization.to_bytes(self.simulation.state))

    def handle_set_state(self, client):
        """Replace current simulation state by the state received from the client

        :param client: client
        """
        client.send(serialization.to_bytes(self.simulation.state))
        updated_state = serialization.from_bytes(self.state, client.recv(self.state_byte_size))
        with self.sim_lock:
            if not self.simulation.paused:
                self.simulation.pause()
                self.simulation.state = updated_state
                self.simulation.resume()
            else:
                self.simulation.state = updated_state

    def handle_pause(self):
        """Pause the simulation
        """
        with self.sim_lock:
            self.simulation.pause()
        logger.info("Simulation paused")

    def handle_resume(self):
        """Resume the simulation
        """
        with self.sim_lock:
            self.simulation.resume()
        logger.info("Simulation resumed")

    def handle_stop(self):
        """Stop the simulation
        """
        with self.sim_lock:
            self.simulation.stop()
        logger.info("Simulation stopped")

    def handle_start(self):
        """Start the simulation
        """
        with self.sim_lock:
            self.simulation.start()
        logger.info("Simulation started")


class Client:
    def __init__(self, server_ip, port, data_size, eval_time):
        self.server_ip = server_ip
        self.port = port
        self.addr = (self.server_ip, self.port)
        self.data_size = data_size
        self.eval_time = eval_time
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.state_example = None
        self.state_bytes_size = None
        self.client_type = None
        self.env = None

    def connect(self):
        """Connect with a remote server and send its client type
        """
        try:
            self.client.connect(self.addr)
            logger.info(f"Connected to {(self.addr, self.port)}")
            self.env_name = self.client.recv(1024).decode()
            logger.info(f"Received {self.env_name} env name")

            self.state_example = pickle.loads(self.client.recv(self.data_size))
            self.state_bytes_size = len(serialization.to_bytes(self.state_example))

            self.client.send(self.client_type.encode())
            self.Env = ENVS[self.env_name]
            time.sleep(1)
        except socket.error as e:
                logger.error(f"error: {e}")

class ReceiveClient(Client):
    def __init__(self, server_ip, port, data_size, eval_time):
        super().__init__(server_ip, port, data_size, eval_time)
        self.client_type = "RECEIVE"

    def receive_loop(self):
        """Start a loop where the client continuously receives and plots new states of the simulation server
        """
        i = 0
        while True:
            try:
                i += 1
                raw_data = self.client.recv(self.state_bytes_size)
                state = serialization.from_bytes(self.state_example, raw_data)
                self.Env.render(state)
            except socket.error as e:
                logger.error(e)
                self.client.close()
                break

    def test(self):
        """Test how many states are received per seconds form the server
        """
        start = time.time()
        i = 0
        while time.time() < start + self.eval_time:
            i += 1
            raw_data = self.client.recv(self.state_bytes_size)
            state = serialization.from_bytes(self.state_example, raw_data)
            self.env.render(state)
        self.client.close()
        logger.info(f"{i=} : {i / self.eval_time} data received per second")


class NotebookClient(Client):
    def __init__(self, server_ip, port, data_size, eval_time):
        super().__init__(server_ip, port, data_size, eval_time)
        self.client_type = "NOTEBOOK"

    def get_state(self):
        """Get the current state of the server's simulation 

        :return: state
        """
        self.client.send("GET_STATE".encode())
        raw_data = self.client.recv(self.state_bytes_size)
        return serialization.from_bytes(self.state_example, raw_data)

    def pause(self):
        """Send a request to pause the server's simulation 
        """
        self.client.send("PAUSE".encode())

    def resume(self):
        """Send a request to resume the server's simulation 
        """
        self.client.send("RESUME".encode())

    def stop(self):
        """Send a request to stop the server's simulation 
        """
        self.client.send("STOP".encode())

    def start(self):
        """Send a request to start the server's simulation 
        """
        self.client.send("START".encode())

    def close(self):
        """Send a request to close the server's simulation 
        """
        self.client.send("CLOSE_CONNECTION".encode())