import time 

import jax
import jax.numpy as jnp
from jax import random

from simulationsandbox.two_d_simulation import SimpleSimulation
from simulationsandbox.three_d_simulation import ThreeDSimulation

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

    sim = SimpleSimulation(MAX_AGENTS, GRID_SIZE)
    state = sim.init_state(NUM_AGENTS, NUM_OBS, key)

    assert sim.max_agents == MAX_AGENTS
    assert sim.grid_size == GRID_SIZE
    assert state.x_pos.shape == (MAX_AGENTS,)
    assert jnp.sum(state.alive) == NUM_AGENTS

