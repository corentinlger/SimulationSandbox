import threading
import time

from jax import random

from simulationsandbox.two_d_simulation import SimpleSimulation

# Independant of any simulation environment 
class SimulationWrapper:
    """
    Simulation wrapper to start, pause, resume and stop a simulation.
    Independant of any simulation type.
    """
    def __init__(self, simulation, state, key):
        self.running = False
        self.paused = False
        self.stop_requested = False
        self.update_thread = None
        # simulation_dependent
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
            print(f"{self.state = }")

            time.sleep(0.1)

    def _update_simulation(self):
        self.key, a_key, step_key = random.split(self.key, 3)
        actions = self.simulation.choose_action(self.state.obs, a_key)
        return self.simulation.step(self.state, actions, step_key)

NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
SLEEP_TIME = 5
SEED = 0

key = random.PRNGKey(SEED)
env = SimpleSimulation(MAX_AGENTS, GRID_SIZE)
state = env.init_state(NUM_AGENTS, NUM_OBS, key)

# Example usage:
sim = SimulationWrapper(env, state, key)

print('Started')
sim.start_simulation()
time.sleep(SLEEP_TIME)

sim.pause_simulation()
print('Paused')
time.sleep(SLEEP_TIME)

print('Resumed')
sim.resume_simulation()
time.sleep(SLEEP_TIME)

sim.stop_simulation()
print('stopped')
