import time 

import jax.numpy as jnp
from jax import random

from simulationsandbox.utils.envs import ENVS

MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
STEP_DELAY = 0.000001
SEED = 0

def test_simple_simulation_run():
    for Env in ENVS.values():
        env = Env(max_agents=MAX_AGENTS, grid_size=GRID_SIZE)
        state = env.init_state(seed=SEED)
        key = random.PRNGKey(SEED)


        # Launch a simulation
        print("Simulation started")
        for timestep in range(NUM_STEPS):
            time.sleep(STEP_DELAY)
            key, step_key = random.split(key)

            if timestep % 10 == 0:
                print(f"step {timestep}")

            if timestep == 20:
                state = env.add_agent(state, 7)
                state = env.add_agent(state, 9)
                state = env.add_agent(state, 5)

            if timestep == 40:
                state = env.remove_agent(state, 2)
                state = env.remove_agent(state, 1)
                state = env.remove_agent(state, 4)

            state = env.step(state, step_key)

            if VIZUALIZE:
                Env.render(state)
        print("\nSimulation ended")

        assert env
        assert state
