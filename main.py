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
    visualize = cfg.params.visualize
    viz_delay = cfg.params.viz_delay

    key = random.PRNGKey(cfg.params.random_seed)

    sim = Simulation(max_agents, grid_size)
    grid = sim.init_grid(grid_size)
    agents_pos, agents_states, num_agents = sim.init_agents(num_agents, max_agents, key)

    # Launch a simulation
    print("\nSimulation started")

    for step in range(num_steps):
        key, a_key, add_key = random.split(key, 3)

        if step % 10 == 0:
            print(f"step {step}")
        
        if step == 20:
            for _ in range(8):
                agents_pos, agents_states, num_agents = sim.add_agent(agents_pos, agents_states, num_agents, add_key)
            
        if step == 40:
            for _ in range(4):
                num_agents = sim.remove_agent(num_agents)

        agents_pos = sim.move_agents(agents_pos, grid_size, a_key)
        agents_states += 0.1

        if visualize:
            sim.visualize(grid, agents_pos, num_agents, viz_delay)

    print("\nSimulation ended")

if __name__ == "__main__":
    main()
