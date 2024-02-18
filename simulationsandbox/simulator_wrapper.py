import threading
import time

from jax import random

class SimulationWrapper:

    def __init__(self, simulation, state, key, step_delay=0.1, update_event=None, sim_lock=None, print_data=False):
        self.running = False
        self.paused = False
        self.stop_requested = False
        self.update_thread = None
        self.print_data = print_data
        self.step_delay = step_delay
        self.update_event = update_event
        # self.sim_lock = sim_lock
        self.simulation = simulation
        self.state = state
        self.key = key
        
    def start(self):
        if not self.running:
            self.running = True
            self.stop_requested = False
            self.update_thread = threading.Thread(target=self.simulation_loop)
            self.update_thread.start()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stop_requested = True

    def simulation_loop(self):
        while not self.stop_requested:
            if self.paused:
                time.sleep(0.1)
                continue

            self.state = self._update_simulation()

            if self.update_event:
                self.update_event.set()

            if self.print_data:
                # print(f"{self.state = }")
                print("stepped")

            time.sleep(self.step_delay) 

    def _update_simulation(self):
        self.key, a_key, step_key = random.split(self.key, 3)
        actions = self.simulation.choose_action(self.state.obs, a_key)
        return self.simulation.step(self.state, actions, step_key)
