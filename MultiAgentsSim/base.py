from functools import partial

from jax import jit
from flax import struct


@struct.dataclass
class SimState:
    time: int
   


class Simulation:
    def __init__(self):
        raise(NotImplementedError)
    

    @partial(jit, static_argnums=(0, 1, 2))
    def init_state(self) -> SimState:
        raise(NotImplementedError)
    

    # Should be moved in another class
    @partial(jit, static_argnums=(0,))
    def choose_action(self, obs):
        raise(NotImplementedError)
    

    # Should also return new obs for agents
    @partial(jit, static_argnums=(0,))
    def step(self, state: SimState, actions, key) -> SimState:
        raise(NotImplementedError)
    

    def get_env_params(self):
        raise(NotImplementedError)


    @staticmethod
    def encode(state):
        raise(NotImplementedError)

    @staticmethod
    def decode(state):
        raise(NotImplementedError)

    @staticmethod
    def visualize(state, color):
            raise(NotImplementedError)


    
