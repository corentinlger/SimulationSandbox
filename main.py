import hydra
from omegaconf import DictConfig, OmegaConf
from jax import random

from simulation import Simulation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    num_agents = cfg.params.num_agents
    max_agents = cfg.params.max_agents
    grid_size = cfg.params.grid_size
    num_steps = cfg.params.num_steps
    rng_key = random.PRNGKey(cfg.params.random_seed)

    sim = Simulation(num_agents, max_agents,grid_size, rng_key)

    print("\nSimulation started")
    grid, agents_pos, agents_states, key = sim.get_env_state()
    grid, final_agent_positions, final_agent_states = sim.simulate(
        grid, agents_pos, agents_states, num_steps, grid_size, key
    )
    print("\nSimulation ended")


if __name__ == "__main__":
    main()
