import jax 
import jax.numpy as jnp 
from jax import random 

import matplotlib.pyplot as plt 

# Will change this later : take the habit of using less classes and more pure functions without self argument 
class Simulation:
    def __init__(self, num_agents, grid_size, key):
        self.grid = self.init_grid(grid_size)
        self.agents_pos, self.agents_states = self.init_agents(num_agents, grid_size, key)
        self.key = key 

    def init_grid(self, grid_size):
        return jnp.zeros((grid_size, grid_size), dtype=jnp.float32)

    def init_agents(self, num_agents, grid_size, key):
        agents_pos = random.randint(key, (num_agents, 2), 0, grid_size)
        agent_states = jnp.ones((num_agents,), dtype=jnp.float32)
        return agents_pos, agent_states

    def move_agents(self, agents_pos, grid_size, key):
        agents_pos += random.randint(key, agents_pos.shape, -1, 2)
        return jnp.clip(agents_pos, 0, grid_size - 1)

    def visualize(self, grid, agents_pos):
        pass

    def simulate(self, grid, agents_pos, agents_states, num_steps, grid_size, key):
        # use a fori_loop after 
        for step in range(num_steps):
            key, a_key = random.split(key)
            agents_pos = self.move_agents(agents_pos, grid_size, a_key)

            agents_states += 0.1

            self.visualize(grid, agents_pos)

        return grid, agents_pos, agents_states
    
    def get_env_state(self):
        return self.grid, self.agents_pos, self.agents_states, self.key


