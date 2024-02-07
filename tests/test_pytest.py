import time 

import jax
import jax.numpy as jnp
from jax import random

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.agents import Agents


NUM_AGENTS = 5 
MAX_AGENTS = 10
NUM_OBS = 3 
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
STEP_DELAY = 0.000001
SEED = 0


def test_simulation_init():
    key = random.PRNGKey(SEED)
    color = (1.0, 0.0, 0.0)

    sim = Simulation(MAX_AGENTS, GRID_SIZE)
    state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

    assert sim.max_agents == MAX_AGENTS
    assert sim.grid_size == GRID_SIZE
    assert state.x_pos.shape == (MAX_AGENTS,)
    assert jnp.sum(state.alive) == NUM_AGENTS
    assert state.grid.shape == (GRID_SIZE, GRID_SIZE)


def test_simulation_run():
    key = random.PRNGKey(SEED)
    color = (1.0, 0.0, 0.0)

    sim = Simulation(MAX_AGENTS, GRID_SIZE)
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

        if timestep == 40:
            state = sim.remove_agent(state, 2)
            state = sim.remove_agent(state, 1)
            state = sim.remove_agent(state, 4)

        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)

        if VIZUALIZE:
            Simulation.visualize_sim(state, color)
    print("\nSimulation ended")
    
    assert jnp.sum(state.alive) == 5
    assert state.x_pos.shape == (MAX_AGENTS,)
    assert state.time == NUM_STEPS

