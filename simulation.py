import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt

# TODO : CHANGE ALL THE FILE SO WE EITHER USE SELF.AGENT8POS OR JUST USE IT IN THE SIMULATION BUTT NOT BOTH

# Will change this later : take the habit of using less classes and more pure functions without self argument
class Simulation:
    def __init__(self, num_agents, max_agents, grid_size, key):
        self.grid = self.init_grid(grid_size)
        self.grid_size = grid_size
        self.max_agents = max_agents
        self.agents_pos, self.agents_states, self.num_agents = self.init_agents(
            num_agents, max_agents, grid_size, key
        )
        self.key = key

    def init_grid(self, grid_size):
        return jnp.zeros((grid_size, grid_size), dtype=jnp.float32)

    def init_agents(self, num_agents, max_agents, grid_size, key):
        if num_agents > max_agents:
            raise(ValueError("num_agents cannot exceed max_agents"))
        
        # set all pos and states to default values in arrays of size max_agents 
        agents_pos = jnp.zeros((max_agents, 2), dtype=jnp.int32)
        agents_states = jnp.zeros((max_agents,), dtype=jnp.float32)
        
        # only update the ones of agents that actually exist
        agents_pos = agents_pos.at[:num_agents].set(random.randint(key, (num_agents, 2), 0, grid_size))
        agents_states = agents_states.at[:num_agents].set(jnp.ones((num_agents,), dtype=jnp.float32))
        
        return agents_pos, agents_states, num_agents

    # TODO : Only move existing agents
    def move_agents(self, agents_pos, grid_size, key):
        agents_pos += random.randint(key, agents_pos.shape, -1, 2)
        return jnp.clip(agents_pos, 0, grid_size - 1)
    
    def add_agent(self, agents_pos, agents_states):
        if self.num_agents < self.max_agents:
            agents_pos = agents_pos.at[self.num_agents].set(*random.randint(self.key, (1, 2), 0, self.grid_size))
            agents_states = agents_states.at[self.num_agents].set(1)
            self.num_agents += 1
            print(f"Added agent {self.num_agents}")
            
        else:
            print("Impossible to add more agents")

        return agents_pos, agents_states


    # TODO:
    def remove_agent(idx=None):
        pass

    def visualize(self, grid, agents_pos, delay=0.1):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))

        plt.clf()

        plt.imshow(grid, cmap="viridis", origin="upper")
        plt.scatter(
            agents_pos[:self.num_agents, 0], agents_pos[:self.num_agents, 1], color="red", marker="o", label="Agents"
        )
        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.draw()
        plt.pause(delay)


    def get_env_state(self):
        return self.grid, self.agents_pos, self.agents_states, self.key
