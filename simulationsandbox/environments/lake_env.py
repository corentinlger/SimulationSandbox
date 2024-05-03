from functools import partial

import jax.numpy as jnp
from jax import random, jit, vmap
from flax import struct
import matplotlib.pyplot as plt

from simulationsandbox.environments.base_env import BaseEnv, BaseEnvState

N_DIMS = 2
GROUND_SPEED = .1
LAKE_SPEED = .05

@struct.dataclass
class Agents:
    pos: jnp.array
    speed: jnp.array
    theta: jnp.array
    alive: jnp.array
    color: jnp.array
    obs: jnp.array


class Object:
    pos: jnp.array


@struct.dataclass
class LakeEnvState(BaseEnvState):
    time: int
    grid_size: int
    lake_pos : jnp.array
    agents: Agents


def normal(theta):
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = jit(vmap(normal))


# Change the angle of the agent a bit 
def turn(obs, key):
    return random.normal(key, shape=()) / 10 

turn = jit(vmap(turn, in_axes=(0, 0)))


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

    def init_state(self, num_agents=5, num_obs=2, seed=0):
        key = random.PRNGKey(seed)
        agents_key_pos, agents_key_theta, lake_key = random.split(key, 3)

        agents = Agents(
            pos=random.uniform(key=agents_key_pos, shape=(self.max_agents, 2), minval=0, maxval=self.grid_size),
            speed=jnp.full(shape=(self.max_agents,), fill_value=GROUND_SPEED),
            theta=random.uniform(key=agents_key_theta, shape=(self.max_agents,), minval=0, maxval=2*jnp.pi),
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
        in_lake = is_point_in_lake(state.agents.pos, state.lake_pos)
        speeds = jnp.where(in_lake, LAKE_SPEED, GROUND_SPEED)
        keys = random.split(key, self.max_agents)
        # Compute the next move of agents
        actions = turn(state.agents.obs, keys)
        theta = state.agents.theta + actions
        n = normal(theta)
        agents_pos = state.agents.pos + (n * jnp.stack((speeds, speeds), axis=1))
        # Collide with walls
        theta = jnp.where(agents_pos[:, 0] < 0, theta - jnp.pi, theta)
        theta = jnp.where(agents_pos[:, 0] > self.grid_size, theta - jnp.pi, theta)
        theta = jnp.where(agents_pos[:, 1] < 0, theta - jnp.pi, theta)
        theta = jnp.where(agents_pos[:, 1] > self.grid_size, theta - jnp.pi, theta)
        # Update new state
        time = state.time + 1
        agents = state.agents.replace(pos=agents_pos, speed=speeds, theta=theta)
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
    def render(state):
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
        plt.pause(0.00001)
