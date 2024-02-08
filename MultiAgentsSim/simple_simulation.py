from functools import partial

import jax.numpy as jnp
from jax import random, jit
from flax import struct
import matplotlib.pyplot as plt

from MultiAgentsSim.base import Simulation, SimState

N_DIMS = 2

# TODO : Add colors
@struct.dataclass
class SimpleSimState(SimState):
    time: int
    grid: jnp.array
    alive: jnp.array
    x_pos: jnp.array
    y_pos: jnp.array
    obs: jnp.array


class SimpleSimulation(Simulation):
    def __init__(self, max_agents, grid_size):
        self.max_agents = max_agents
        self.grid_size = grid_size


    @partial(jit, static_argnums=(0, 1, 2))
    def init_state(self, num_agents, num_obs, key):
        x_key, y_key = random.split(key)
        return SimpleSimState(time=0,
                        grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.float32),
                        alive = jnp.hstack((jnp.ones(num_agents), jnp.zeros(self.max_agents - num_agents))),
                        x_pos=random.randint(key=x_key, shape=(self.max_agents,), minval=0, maxval=self.grid_size),
                        y_pos=random.randint(key=y_key, shape=(self.max_agents,), minval=0, maxval=self.grid_size),
                        obs = jnp.zeros((self.max_agents, num_obs))
                        )
    

    @partial(jit, static_argnums=(0,))
    def choose_action(self, obs, key):
        return random.randint(key, shape=(obs.shape[0], N_DIMS), minval=-1, maxval=2)
    

    @partial(jit, static_argnums=(0,))
    def step(self, sim_state, actions, key):
        x_pos = jnp.clip(sim_state.x_pos + actions[:, 0], 0, self.grid_size - 1)
        y_pos = jnp.clip(sim_state.y_pos + actions[:, 1], 0, self.grid_size - 1)
        time = sim_state.time + 1
        sim_state = sim_state.replace(time=time, x_pos=x_pos, y_pos=y_pos)
        return sim_state
    

    # TODO : See if I had if conditions to check wether the idx is valid or not, or if I jit the function
    def add_agent(self, sim_state, agent_idx):
        sim_state = sim_state.replace(alive=sim_state.alive.at[agent_idx].set(1.0))
        print(f"agent {agent_idx} added")
        return sim_state
    
    
    def remove_agent(self, sim_state, agent_idx):
        sim_state = sim_state.replace(alive=sim_state.alive.at[agent_idx].set(0.0))
        print(f"agent {agent_idx} removed")
        return sim_state
    

    def get_env_params(self):
        return self.grid_size, self.max_agents

    @staticmethod
    def encode(state):
        pass

    @staticmethod
    def decode(state):
        pass

    @staticmethod
    def visualize_sim(state, color, grid_size=None):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))

        plt.clf()

        plt.imshow(state.grid, cmap="viridis", origin="upper")

        alive_agents = jnp.where(state.alive != 0.0)
        agents_x_pos = state.x_pos[alive_agents]
        agents_y_pos = state.y_pos[alive_agents]
        plt.scatter(
            agents_x_pos, agents_y_pos, color=color, marker="o", label="Agents"
        )
        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.draw()
        plt.pause(0.001)


    
