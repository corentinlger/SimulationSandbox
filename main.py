import time 

import hydra
from omegaconf import DictConfig, OmegaConf
from jax import random

from MultiAgentsSim.simulation import Simulation


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

    key = random.PRNGKey(cfg.params.random_seed)
    color = (1.0, 0.0, 0.0)

    sim = Simulation(max_agents, grid_size)
    state = sim.init_state(num_agents, num_obs, key)


    # Launch a simulation
    print("Simulation started")
    for timestep in range(num_steps):
        time.sleep(step_delay)
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

        if visualize:
            Simulation.visualize_sim(state, color)
    print("\nSimulation ended")

if __name__ == "__main__":
    main()
