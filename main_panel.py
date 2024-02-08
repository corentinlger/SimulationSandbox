import time 

import hydra
from omegaconf import DictConfig, OmegaConf
from jax import random

from MultiAgentsSim.simple_simulation import SimpleSimulation
from MultiAgentsSim.three_d_simulation import ThreeDSimulation

import panel as pn
from param import Parameterized, Parameter

import threading as tr

pn.extension()

class Visualizer(Parameterized):
    simulation = Parameter()
    state = Parameter()
    color = Parameter()

    def __init__(self, sim, state, color, **params):
        super().__init__(**params)
        self.simulation = sim
        self.state = state
        self.color = color
    
    @pn.depends("state", "color")
    def visualize(self):
        return self.simulation.visualize_panel(self.state, self.color)
    
def update_sim(num_steps, step_delay, key, sim, v, state):
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
        
        v.state = state
        # if visualize:
        #     Simulation.visualize_sim(state, color, grid_size)
    print("\nSimulation ended")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    global num_steps, step_delay, key, sim, v
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
    color = (1.0, 0.0, 0.0)

    # Choose a simulation type
    if sim_type == "simple":
        Simulation = SimpleSimulation
    elif sim_type == "three_d":
        Simulation = ThreeDSimulation
    else:
        raise(ValueError(f"Unknown sim type {sim_type}"))

    sim = Simulation(max_agents, grid_size)
    state = sim.init_state(num_agents, num_obs, key)

    v = Visualizer(sim, state, color)

    win = pn.Row(f"Simulation {Simulation.__name__}", v.visualize)
    win.servable()

    update_thread = tr.Thread(target=update_sim, daemon=True, args=(num_steps, step_delay, key, sim, v, state))
    update_thread.start()

    # # Launch a simulation
    # print("Simulation started")
    # for timestep in range(num_steps):
    #     time.sleep(step_delay)
    #     key, a_key, step_key = random.split(key, 3)

    #     if timestep % 10 == 0:
    #         print(f"step {timestep}")
    #         print(f"{state.x_pos = }")
    #         print(f"{state.y_pos = }")

    #     if timestep == 20:
    #         state = sim.add_agent(state, 7)
    #         state = sim.add_agent(state, 9)
    #         state = sim.add_agent(state, 5)

    #     if timestep == 40:
    #         state = sim.remove_agent(state, 2)
    #         state = sim.remove_agent(state, 1)
    #         state = sim.remove_agent(state, 4)

    #     actions = sim.choose_action(state.obs, a_key)
    #     state = sim.step(state, actions, step_key)
        
    #     v.state = state
    #     # if visualize:
    #     #     Simulation.visualize_sim(state, color, grid_size)
    # print("\nSimulation ended")

main()
