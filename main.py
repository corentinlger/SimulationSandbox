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

    rng_key = random.PRNGKey(cfg.params.random_seed)

    sim = Simulation(num_agents, max_agents, grid_size, rng_key)

    # Launch a simulation
    print("\nSimulation started")

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

        if step == 40:
             sim.remove_agent()
             sim.remove_agent()
             sim.remove_agent()
             sim.remove_agent()

        key, a_key = random.split(key)

        agents_pos = sim.move_agents(agents_pos, grid_size, a_key)
        agents_states += 0.1

        if visualize:
            sim.visualize(grid, agents_pos, viz_delay)

    print("\nSimulation ended")

if __name__ == "__main__":
    main()
