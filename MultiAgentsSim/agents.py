from functools import partial

import jax.numpy as jnp 
from jax import random, jit
from flax import struct


@struct.dataclass
class AgentsState:
    alive: jnp.array
    x_pos: jnp.array
    y_pos: jnp.array
    obs: jnp.array


class Agents:
    def __init__(self, max_agents):
        self.max_agents = max_agents


    def init_agents(self, num_agents, max_agents, num_obs, grid_size, key):
        return AgentsState(alive = jnp.hstack((jnp.ones(num_agents), jnp.zeros(max_agents - num_agents))),
                           x_pos=random.randint(key=key, shape=(max_agents,), minval=0, maxval=self.gr),
                           y_pos=random.randint(key=key, shape=(max_agents,), minval=0, maxval=grid_size),
                           obs = jnp.zeros((max_agents, num_obs))
                           )


    


