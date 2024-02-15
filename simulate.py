import time 

import hydra
from omegaconf import DictConfig, OmegaConf
from jax import random

from simulationsandbox.two_d_simulation import SimpleSimulation
from simulationsandbox.three_d_simulation import ThreeDSimulation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    num_agents = cfg.params.num_agents
    max_agents = cfg.params.max_agents
    num_obs = cfg.params.num_obs
    grid_size = cfg.params.grid_size
    num_steps = cfg.params.num_steps
    visualize = cfg.params.visualize
    step_delay = cfg.params.step_delay
    sim_type = cfg.params.sim_type

    key = random.PRNGKey(cfg.params.random_seed)

    # Choose a simulation type
    if sim_type == "two_d":
        Simulation = SimpleSimulation
    elif sim_type == "three_d":
        Simulation = ThreeDSimulation
    else:
        raise(ValueError(f"Unknown sim type {sim_type}"))

    sim = Simulation(max_agents, grid_size)
    state = sim.init_state(num_agents, num_obs, key)

    # Launch a simulation
    print("Simulation started")
    for timestep in range(num_steps):
        time.sleep(step_delay)
        key, a_key, step_key = random.split(key, 3)

        if timestep % 10 == 0:
            print(f"\nstep {timestep}")

        if timestep == (num_steps // 3):
            # Add 3 agents and change the color of an agent
            state = sim.add_agent(state, 7)
            state = sim.add_agent(state, 9)
            state = sim.add_agent(state, 5)
            state = state.replace(colors=state.colors.at[0, 2].set(1.0))

        if timestep ==  2* (num_steps // 3):
            # Remove 3 other agents and change the color of another agent
            state = sim.remove_agent(state, 2)
            state = sim.remove_agent(state, 1)
            state = sim.remove_agent(state, 4)
            state = state.replace(colors=state.colors.at[7, 1].set(1.0))


        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)

        if visualize:
            Simulation.visualize_sim(state)
    print("\nSimulation ended")

if __name__ == "__main__":
    main()
