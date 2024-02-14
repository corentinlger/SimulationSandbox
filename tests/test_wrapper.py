import time 

import jax
import jax.numpy as jnp
from jax import random

from simulationsandbox.two_d_simulation import SimpleSimulation
from simulationsandbox.three_d_simulation import ThreeDSimulation
from simulationsandbox.wrapper import SimulationWrapper

NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
SLEEP_TIME = 2
STEP_DELAY = 0.01
PRINT_DATA = True
SEED = 0


def test_wrapper_two_d_sim(): 
    key = random.PRNGKey(SEED)
    env = SimpleSimulation(MAX_AGENTS, GRID_SIZE)
    state = env.init_state(NUM_AGENTS, NUM_OBS, key)

    # Example usage:
    sim = SimulationWrapper(env, state, key, step_delay=STEP_DELAY, print_data=PRINT_DATA)

    print('Started')
    sim.start()
    time.sleep(SLEEP_TIME)
    assert sim.running == True

    sim.pause()
    print('Paused')
    time.sleep(SLEEP_TIME)
    assert sim.paused == True

    print('Resumed')
    sim.resume()
    time.sleep(SLEEP_TIME)
    assert sim.running == True

    sim.stop()
    print('stopped')


# def test_wrapper_three_d_sim(): 
#     key = random.PRNGKey(SEED)
#     env = ThreeDSimulation(MAX_AGENTS, GRID_SIZE)
#     state = env.init_state(NUM_AGENTS, NUM_OBS, key)

#     # Example usage:
#     sim = SimulationWrapper(env, state, key, print_data=True)

#     print('Started')
#     sim.start()
#     time.sleep(SLEEP_TIME)
#     assert sim.running == True

#     sim.pause()
#     print('Paused')
#     time.sleep(SLEEP_TIME)
#     assert sim.paused == True

#     print('Resumed')
#     sim.resume()
#     time.sleep(SLEEP_TIME)
#     assert sim.running == True

#     sim.stop()
#     print('stopped')