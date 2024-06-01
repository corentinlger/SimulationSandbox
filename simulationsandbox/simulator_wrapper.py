import threading
import time

from jax import random

class SimulationWrapper:
    def __init__(self, env, state, seed, step_delay=0.1, update_event=None, sim_lock=None, print_data=False):
        self.running = False
        self.paused = False
        self.stop_requested = False
        self.update_thread = None
        self.print_data = print_data
        self.step_delay = step_delay
        self.update_event = update_event
        # self.sim_lock = sim_lock
        self.env = env
        self.state = state
        self.key = random.PRNGKey(seed)
        
    def start(self):
        """Start the simulation
        """
        if not self.running:
            self.running = True
            self.stop_requested = False
            self.update_thread = threading.Thread(target=self.simulation_loop)
            self.update_thread.start()

    def pause(self):
        """Pause the simulation
        """
        self.paused = True

    def resume(self):
        """Resume the simulation
        """
        self.paused = False

    def stop(self):
        """Stop the simulation
        """
        self.stop_requested = True

    def simulation_loop(self):
        """Start a simulation loop that updates the simulation state at regular intervals, unless the simulation is paused
        """
        while not self.stop_requested:
            if self.paused:
                time.sleep(0.1)
                continue

            self.state = self._update_env_state()

            if self.update_event:
                self.update_event.set()

            if self.print_data:
                print("stepped")

            time.sleep(self.step_delay) 

    def _update_env_state(self):
        """Update the environment state

        :return: new state 
        """
        self.key, key = random.split(self.key)
        return self.env.step(self.state, key)
