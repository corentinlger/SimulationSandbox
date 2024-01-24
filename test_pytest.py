import jax

from simulation import Simulation


def test_simulation_init():
    num_agents = 5
    grid_size = 20
    key = jax.random.PRNGKey(0)

    sim = Simulation(num_agents=num_agents, grid_size=grid_size, key=key)

    assert sim.agents_pos.shape == (num_agents, 2)
    assert sim.grid.shape == (grid_size, grid_size)


def test_simulation_run():
    num_agents = 5
    grid_size = 20
    key = jax.random.PRNGKey(0)
    num_steps = 100

    sim = Simulation(num_agents=num_agents, grid_size=grid_size, key=key)

    grid, agents_pos, agents_states, key = sim.get_env_state()
    grid, final_agent_positions, final_agent_states = sim.simulate(
        grid, agents_pos, agents_states, num_steps, grid_size, key, visualize=False
    )

    assert final_agent_positions.shape == (num_agents, 2)
    assert final_agent_states.shape == (num_agents,)
