from functools import partial

import jax.numpy as jnp 
from jax import random, jit

class Agents:
    def __init__(self, max_agents, grid_size):
        self.max_agents = max_agents
        self.grid_size = grid_size

    def init_agents(self, num_agents, max_agents, key):
        # set all pos and states to default values in arrays of size max_agents 
        agents_pos = jnp.zeros((max_agents, 2), dtype=jnp.int32)
        agents_states = jnp.zeros((max_agents,), dtype=jnp.float32)
        
        # only update the ones of agents that actually exist
        new_pos = random.randint(key, (num_agents, 2), 0, self.grid_size)
        agents_pos = agents_pos.at[:num_agents].set(new_pos)
        agents_states = agents_states.at[:num_agents].set(jnp.ones((num_agents,), dtype=jnp.float32))
        
        return agents_pos, agents_states, num_agents

    
    @partial(jit, static_argnums=(0,))
    def choose_action(self, agents_pos, key_a):
        # Choose a random action at the moment 
        return random.randint(key_a, agents_pos.shape, -1, 2)


