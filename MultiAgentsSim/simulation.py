from functools import partial

import jax.numpy as jnp
from jax import random, jit
from flax import struct
import matplotlib.pyplot as plt


# import matplotlib
# matplotlib.use('agg')

# TODO : 
@struct.dataclass
class SimState:
    time: int

@struct.dataclass
class SimParams:
    max_agents: int
    grid_size: int 



class Simulation:
    def __init__(self, max_agents, grid_size):
        self.grid_size = grid_size
        self.grid = self.init_grid(grid_size)
        self.max_agents = max_agents

    def init_grid(self, grid_size):
        return jnp.zeros((grid_size, grid_size), dtype=jnp.float32)

    def init_agents(self, num_agents, max_agents, key):
        if num_agents > max_agents:
            raise(ValueError("num_agents cannot exceed max_agents"))
        
        # set all pos and states to default values in arrays of size max_agents 
        agents_pos = jnp.zeros((max_agents, 2), dtype=jnp.int32)
        agents_states = jnp.zeros((max_agents,), dtype=jnp.float32)
        
        # only update the ones of agents that actually exist
        agents_pos = agents_pos.at[:num_agents].set(random.randint(key, (num_agents, 2), 0, self.grid_size))
        agents_states = agents_states.at[:num_agents].set(jnp.ones((num_agents,), dtype=jnp.float32))
        
        return agents_pos, agents_states, num_agents
    
    def choose_random_action(self, agents_pos, key_a):
        return random.randint(key_a, agents_pos.shape, -1, 2)
    
    @partial(jit, static_argnums=(0, ))
    def move_agents(self, agents_pos, agents_movements):
        agents_pos += agents_movements
        return jnp.clip(agents_pos, 0, self.grid_size - 1)
    
    def add_agent(self, agents_pos, agents_states, num_agents, key):
        if num_agents < self.max_agents:
            agents_pos = agents_pos.at[num_agents].set(*random.randint(key, (1, 2), 0, self.grid_size))
            agents_states = agents_states.at[num_agents].set(1)
            num_agents += 1
            print(f"Added agent {num_agents}")
            
        else:
            print("Impossible to add more agents")

        return agents_pos, agents_states, num_agents

    def remove_agent(self, num_agents):
        if num_agents <= 0:
            print("There is no agents to remove")
        else:
            num_agents -= 1
            print(f"Removed agent {num_agents + 1}")
        return num_agents
        
    @staticmethod
    def visualize_sim(grid, agents_pos, num_agents, color):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))

        plt.clf()

        plt.imshow(grid, cmap="viridis", origin="upper")
        plt.scatter(
            agents_pos[:num_agents, 0], agents_pos[:num_agents, 1], color=color, marker="o", label="Agents"
        )
        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.draw()
        plt.pause(0.001)


    @staticmethod
    def visualize_simzz(grid, agents_pos, num_agents, color):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))

        plt.clf()

        plt.imshow(grid, cmap="viridis", origin="upper")
        plt.scatter(
            agents_pos[:num_agents, 0], agents_pos[:num_agents, 1], color=color, marker="o", label="Agents"
        )
        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.draw()
        plt.pause(0.001)


    def get_env_params(self):
        return self.grid_size, self.max_agents
