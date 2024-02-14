import time

from jax import random

from simulationsandbox.two_d_simulation import SimpleSimulation
from simulationsandbox.wrapper import SimulationWrapper

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
sim = SimulationWrapper(env, state, key, print_data=True)

print('Started')
sim.start()
time.sleep(SLEEP_TIME)

sim.pause()
print('Paused')
time.sleep(SLEEP_TIME)

print('Resumed')
sim.resume()
time.sleep(SLEEP_TIME)

sim.stop()
print('stopped')
