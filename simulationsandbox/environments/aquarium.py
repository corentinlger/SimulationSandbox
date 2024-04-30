from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random, jit, vmap
from flax import struct

from simulationsandbox.environments.base_env import BaseEnv, BaseEnvState

N_DIMS = 3
FISH_SPEED = 4.
FISH_COLOR = [0., 0.5, 1.]
MOVE_SCALE = 1.
FOOD_SPEED = 1.
FOOD_COLOR = [0.8, 0.5, 0.]


@struct.dataclass
class Agents:
    pos: jnp.array
    velocity: jnp.array
    alive: jnp.array
    color: jnp.array
    obs: jnp.array


@struct.dataclass
class Objects:
    pos: jnp.array
    velocity: jnp.array
    color: jnp.array
    exist: jnp.array


@struct.dataclass
class AquiariumState(BaseEnvState):
    time: int
    grid_size: int
    agents: Agents
    objects: Objects


def normal(theta):
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = jit(vmap(normal))


# Change the angle and speed of the agent a bit 
def move(obs, key):
    return random.normal(key, shape=(3,)) * MOVE_SCALE

move = jit(vmap(move, in_axes=(0, 0)))


class Aquarium(BaseEnv):
    """ Minimalistic aquarium environmnent"""
    def __init__(self, max_agents=10, max_objects=20, grid_size=50):
        self.max_agents = max_agents
        self.max_objects = max_objects
        self.grid_size = grid_size

    def init_state(self, num_agents, num_obs, key):
        agents_key_pos, agents_key_vel, agents_color_key, objects_key_pos = random.split(key, 4)
        fish_velocity = random.uniform(agents_key_vel, shape=(self.max_agents, N_DIMS), minval=-1, maxval=1)
        fish_velocity = (fish_velocity / jnp.linalg.norm(fish_velocity)) * FISH_SPEED
        # fish_velocity = fish_velocity  * FISH_SPEED

        fish = Agents(
            pos=random.uniform(key=agents_key_pos, shape=(self.max_agents, N_DIMS), minval=0, maxval=self.grid_size),
            velocity=fish_velocity,
            alive=jnp.hstack((jnp.ones(num_agents), jnp.zeros(self.max_agents - num_agents))),
            # color=random.uniform(key=agents_color_key, shape=(self.max_agents, 3), minval=0., maxval=1.),
            color=jnp.full((self.max_agents, 3), jnp.array(FISH_COLOR)),
            obs=jnp.zeros((self.max_agents, num_obs))
        )
        
        # Add food at the surface of the aquarium
        x_y_food_pos=random.uniform(key=objects_key_pos, shape=(self.max_objects, 2), minval=0, maxval=self.grid_size) 
        z_food_pos = jnp.full((self.max_objects, 1), fill_value=self.grid_size)
        food_pos = jnp.concatenate((x_y_food_pos, z_food_pos), axis=1)
        
        food = Objects(
            pos=food_pos,
            velocity=jnp.tile(jnp.array([0., 0., -1]), (self.max_objects, 1)) * FOOD_SPEED,
            color=jnp.full((self.max_objects, 3), jnp.array(FOOD_COLOR)),
            exist=jnp.full((self.max_objects), 1.)
        )

        aquarium_env =  AquiariumState(
            time=0,
            grid_size=self.grid_size,
            agents=fish,
            objects=food
        )
        
        return aquarium_env
    
    @partial(jit, static_argnums=(0,))
    def step(self, state, key):
        # Update agents positions
        keys = random.split(key, self.max_agents)
        d_vel = move(state.agents.obs, keys)
        velocity = state.agents.velocity + d_vel
        velocity = (velocity / jnp.linalg.norm(velocity)) * FISH_SPEED
        agents_pos = state.agents.pos + velocity
        # Collide with walls
        agents_pos = jnp.clip(agents_pos, 0, self.grid_size)
        agents = state.agents.replace(pos=agents_pos, velocity=velocity)

        # Update food position
        food_vel = state.objects.velocity
        food_pos = state.objects.pos + food_vel
        food_pos = jnp.clip(food_pos, 0, self.grid_size)
        objects = state.objects.replace(pos=food_pos)
        
        # Update new state
        time = state.time + 1
        state = state.replace(time=time, agents=agents, objects=objects)
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
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        plt.clf()

        ax = plt.axes(projection='3d')

        alive_agents = jnp.where(state.agents.alive != 0.0)
        agents_x_pos = state.agents.pos[:, 0][alive_agents]
        agents_y_pos = state.agents.pos[:, 1][alive_agents]
        agents_z_pos = state.agents.pos[:, 2][alive_agents]
        agents_colors = state.agents.color[alive_agents]

        exist_object = jnp.where(state.objects.exist != 0.0)
        objects_x_pos = state.objects.pos[:, 0][exist_object]
        objects_y_pos = state.objects.pos[:, 1][exist_object]
        objects_z_pos = state.objects.pos[:, 2][exist_object]
        objects_colors = state.objects.color[exist_object]
        print(objects_x_pos)

        # TODO : see how to add cmap=colormaps["gist_rainbow"] for fish colors
        ax.scatter(agents_x_pos, agents_y_pos, agents_z_pos, c=agents_colors, marker="o", label="Fish")
        ax.scatter(objects_x_pos, objects_y_pos, objects_z_pos, c=objects_colors, marker="o", label="Fish")

        ax.set_title("Multi-Agent Simulation")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_xlim(0, state.grid_size)
        ax.set_ylim(0, state.grid_size)
        ax.set_zlim(0, state.grid_size)

        ax.legend()

        plt.draw()
        plt.pause(0.001)
