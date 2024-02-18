import time 

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt 

from simulationsandbox.environments.two_d_example_env import TwoDEnv
from simulationsandbox.environments.three_d_example_env import ThreeDEnv

NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
STEP_DELAY = 0.000001
SEED = 0


def test_simple_simulation_run():
    key = random.PRNGKey(SEED)
    sim = TwoDEnv(MAX_AGENTS, GRID_SIZE)
    state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

    # Launch a simulation
    print("Simulation started")
    for timestep in range(NUM_STEPS):
        time.sleep(STEP_DELAY)
        key, a_key, step_key = random.split(key, 3)

        if timestep % 10 == 0:
            print(f"step {timestep}")
            print(f"{state.x_pos = }")
            print(f"{state.y_pos = }")

        if timestep == 20:
            state = sim.add_agent(state, 7)
            state = sim.add_agent(state, 9)
            state = sim.add_agent(state, 5)
            state = state.replace(colors=state.colors.at[0, 2].set(1.0))


        if timestep == 40:
            state = sim.remove_agent(state, 2)
            state = sim.remove_agent(state, 1)
            state = sim.remove_agent(state, 4)
            state = state.replace(colors=state.colors.at[7, 1].set(1.0))


        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)

        if VIZUALIZE:
            TwoDEnv.visualize_sim(state)
    print("\nSimulation ended")

    assert jnp.sum(state.alive) == 5
    assert state.x_pos.shape == (MAX_AGENTS,)
    assert state.time == NUM_STEPS



def test_three_d_simulation_run():
    key = random.PRNGKey(SEED)
    sim = ThreeDEnv(MAX_AGENTS, GRID_SIZE)
    state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

    # Launch a simulation
    print("Simulation started")
    for timestep in range(NUM_STEPS):
        time.sleep(STEP_DELAY)
        key, a_key, step_key = random.split(key, 3)

        if timestep % 10 == 0:
            print(f"step {timestep}")
            print(f"{state.x_pos = }")
            print(f"{state.y_pos = }")

        if timestep == 20:
            state = sim.add_agent(state, 7)
            state = sim.add_agent(state, 9)
            state = sim.add_agent(state, 5)
            state = state.replace(colors=state.colors.at[0, 2].set(1.0))

        if timestep == 40:
            
            state = sim.remove_agent(state, 2)
            state = sim.remove_agent(state, 1)
            state = sim.remove_agent(state, 4)
            state = state.replace(colors=state.colors.at[7, 1].set(1.0))

        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)

        if VIZUALIZE:
            ThreeDEnv.visualize_sim(state)

    plt.close()
    print("\nSimulation ended")


