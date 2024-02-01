import jax
from jax import random

from MultiAgentsSim.simulation import Simulation
from MultiAgentsSim.agents import Agents

NUM_AGENTS = 5 
MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
VIZ_DELAY = 0.000001
SEED = 0


def test_simulation_init():
    key = jax.random.PRNGKey(SEED)

    sim = Simulation(max_agents=MAX_AGENTS, grid_size=GRID_SIZE)
    grid = sim.init_grid(GRID_SIZE)
    agents_pos, agents_states, num_agents = sim.init_agents(NUM_AGENTS, MAX_AGENTS, key)

    assert sim.max_agents == MAX_AGENTS
    assert agents_pos.shape == (MAX_AGENTS, 2)
    assert num_agents == NUM_AGENTS
    assert grid.shape == (GRID_SIZE, GRID_SIZE)


def test_simulation_run():
    key = jax.random.PRNGKey(SEED)

    sim = Simulation(MAX_AGENTS, GRID_SIZE)
    agents = Agents(MAX_AGENTS, GRID_SIZE)

    grid = sim.init_grid(GRID_SIZE)
    agents_pos, agents_states, num_agents = agents.init_agents(NUM_AGENTS, MAX_AGENTS, key)

    for step in range(NUM_STEPS):
        key, a_key, add_key = random.split(key, 3)

        if step % 10 == 0:
            print(f"step {step}")
        
        if step == 20:
            for _ in range(8):
                agents_pos, agents_states, num_agents = sim.add_agent(agents_pos, agents_states, num_agents, add_key)
            
        if step == 40:
            for _ in range(4):
                num_agents = sim.remove_agent(num_agents)

        actions = agents.choose_action(agents_pos, a_key)
        agents_pos = sim.move_agents(agents_pos, actions)
        agents_states += 0.1

        if VIZUALIZE:
            Simulation.visualize_sim(grid, agents_pos, num_agents, VIZ_DELAY)

    assert num_agents == 6
    assert agents_pos.shape == (MAX_AGENTS, 2)


# def test_add_remove_agents():
#     key = jax.random.PRNGKey(SEED)

#     sim = Simulation(num_agents=NUM_AGENTS, max_agents=MAX_AGENTS, grid_size=GRID_SIZE, key=key)
#     grid, agents_pos, agents_states, key = sim.get_env_state()

#     assert sim.num_agents == 5

#     for i in range(15):
#         agents_pos, agents_states = sim.add_agent(agents_pos, agents_states)

#     assert sim.num_agents == MAX_AGENTS

#     for i in range(15):
#         sim.remove_agent()
    
#     assert sim.num_agents == 0
