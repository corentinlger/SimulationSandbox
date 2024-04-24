# TODO : Ensure the agents do not spawn in the lake 
# TODO : Change the init of lake 
# TODO : Add more elements to the environment
# TODO : Maybe test particle based with different speeds on earth / lake


from functools import partial

import jax.numpy as jnp
from jax import random, jit, vmap
from flax import struct
import matplotlib.pyplot as plt

from simulationsandbox.environments.base_env import BaseEnv, BaseEnvState

N_DIMS = 2


@struct.dataclass
class Agents:
    pos: jnp.array
    alive: jnp.array
    color: jnp.array
    obs: jnp.array


@struct.dataclass
class LakeEnvState(BaseEnvState):
    time: int
    grid_size: int
    lake_pos : jnp.array
    agents: Agents


def move(obs, key):
    return random.randint(key, shape=(N_DIMS,), minval=-1, maxval=2) / 10

move = jit(vmap(move, in_axes=(0, 0)))


def is_point_in_lake(agent_pos, lake_coordinates):
    on_lake_corners = jnp.any((lake_coordinates == agent_pos).all(axis=1))
    
    crossing_number = 0

    for i in range(len(lake_coordinates)):
        vertex1 = lake_coordinates[i]
        vertex2 = lake_coordinates[(i + 1) % len(lake_coordinates)]

        cond1 = (vertex1[1] > agent_pos[1]) != (vertex2[1] > agent_pos[1])
        cond2 = (agent_pos[0] < (vertex2[0] - vertex1[0]) * (agent_pos[1] - vertex1[1]) / (vertex2[1] - vertex1[1]) + vertex1[0])
        crossing_number = jnp.where(cond1 & cond2, crossing_number + 1, crossing_number)

    in_lake = crossing_number % 2 == 1

    return (on_lake_corners | in_lake)

is_point_in_lake = jit(vmap(is_point_in_lake, in_axes=(0, None)))


class LakeEnv(BaseEnv):
    """ Minimalistic environmnent with a lake in the middle """
    def __init__(self, max_agents, grid_size):
        self.max_agents = max_agents
        self.grid_size = grid_size

    def init_state(self, num_agents, num_obs, key):
        agents_key, lake_key = random.split(key)

        agents = Agents(
            pos=random.randint(key=agents_key, shape=(self.max_agents, 2), minval=0, maxval=self.grid_size),
            alive=jnp.hstack((jnp.ones(num_agents), jnp.zeros(self.max_agents - num_agents))),
            color=jnp.full(shape=(self.max_agents, 3), fill_value=jnp.array([1.0, 0.0, 0.0])),
            obs=jnp.zeros((self.max_agents, num_obs))
        )

        lake_env =  LakeEnvState(
            time=0,
            grid_size=self.grid_size,
            lake_pos=self.create_lake_coordinates(key=lake_key, grid_size=self.grid_size),
            agents=agents
        )
        
        return lake_env
    
    def create_lake_coordinates(self, key, grid_size):
        # Create a lake in the middle of the env
        pos = random.normal(key, shape=(4,2))
        mid = jnp.array([grid_size//2, grid_size//2])
        pos = mid + jnp.int32(pos * grid_size // 3)
        return pos 

    @partial(jit, static_argnums=(0,))
    def step(self, state, key):
        keys = random.split(key, self.max_agents)
        # Compute the next move of agents
        actions = move(state.agents.obs, keys)
        next_positions = state.agents.pos + actions
        # Keep agents in the grid
        agents_pos = jnp.clip(next_positions, 0, self.grid_size - 1)
        # Keep agents outside the lake
        move_in_lake = is_point_in_lake(next_positions, state.lake_pos)
        move_in_lake = jnp.stack((move_in_lake, move_in_lake), axis=1)
        agents_pos = jnp.where(move_in_lake, agents_pos, next_positions)
        # Update new state
        time = state.time + 1
        agents = state.agents.replace(pos=agents_pos)
        state = state.replace(time=time, agents=agents)
        return state
    
    def add_agent(self, state, agent_idx):
        agents = state.agents.replace(alive=state.agents.alive.at[agent_idx].set(1.0))
        state = state.replace(agents=agents)
        return state
    
    def remove_agent(self, state, agent_idx):
        agents = state.agents.replace(alive=state.agents.alive.at[agent_idx].set(0.0))
        state = state.replace(agents=agents)
        return state

    @staticmethod
    def visualize_sim(state):
        if not plt.fignum_exists(1):
            plt.ion()
            plt.figure(figsize=(10, 10))
        plt.clf()

        # Draw lake
        first_pos = state.lake_pos[0].reshape(1, 2)
        lake_pos = jnp.append(state.lake_pos, first_pos, axis=0)
        x = lake_pos[:, 0]
        y = lake_pos[:, 1]
        plt.fill(x, y, color='lightblue')

        # Draw agents
        alive_agents = jnp.where(state.agents.alive != 0.0)
        agents_x_pos = state.agents.pos[:, 0][alive_agents]
        agents_y_pos = state.agents.pos[:, 1][alive_agents]
        agents_colors = state.agents.color[alive_agents]
        plt.scatter(
            agents_x_pos, agents_y_pos, c=agents_colors, marker="o", label="Agents"
        )

        plt.title("Multi-Agent Simulation")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.xlim(0, state.grid_size)
        plt.ylim(0, state.grid_size)

        plt.draw()
        plt.pause(0.0001)