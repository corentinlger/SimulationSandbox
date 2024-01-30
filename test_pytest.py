import jax
from jax import random

from simulation import Simulation

NUM_AGENTS = 5 
MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
VIZ_DELAY = 0.001
SEED = 0


def test_simulation_init():
    key = jax.random.PRNGKey(SEED)

    sim = Simulation(num_agents=NUM_AGENTS, max_agents=MAX_AGENTS, grid_size=GRID_SIZE, key=key)

    assert sim.num_agents == NUM_AGENTS
    assert sim.agents_pos.shape == (MAX_AGENTS, 2)
    assert sim.grid.shape == (GRID_SIZE, GRID_SIZE)


def test_simulation_run():
    key = jax.random.PRNGKey(SEED)
    num_steps = 100

    sim = Simulation(num_agents=NUM_AGENTS, max_agents=MAX_AGENTS, grid_size=GRID_SIZE, key=key)

    assert sim.num_agents == 5

    grid, agents_pos, agents_states, key = sim.get_env_state()

    for step in range(num_steps):
        if step % 10 == 0:
            print(f"step {step}")
        
        if step == 20:
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)
                agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)

        key, a_key = random.split(key)

        agents_pos = sim.move_agents(agents_pos, GRID_SIZE, a_key)
        agents_states += 0.1

        if VIZUALIZE:
            sim.visualize(grid, agents_pos, VIZ_DELAY)

    assert sim.num_agents == MAX_AGENTS
    assert agents_pos.shape == (MAX_AGENTS, 2)