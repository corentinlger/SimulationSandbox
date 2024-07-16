from functools import partial
from functools import reduce

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random, jit, vmap
from flax import struct

from simulationsandbox.environments.base_env import BaseEnv, BaseEnvState

# Env
N_DIMS = 3
# Fish
FISH_SPEED = 4.
FISH_COLOR = [0., 0.5, 1.]
FISH_EATING_RANGE = 3.
MOVE_SCALE = 1.
# Food
FOOD_SPEED = 1.
FOOD_COLOR = [0.8, 0.5, 0.]


@struct.dataclass
class Agents:
    pos: jnp.array
    velocity: jnp.array
    alive: jnp.array
    color: jnp.array
    obs: jnp.array
    energy: jnp.array


@struct.dataclass
class Objects:
    pos: jnp.array
    velocity: jnp.array
    color: jnp.array
    exist: jnp.array


# TODO : Add key in the state
@struct.dataclass
class AquiariumState(BaseEnvState):
    time: int
    grid_size: int
    agents: Agents
    objects: Objects


# TODO : see if we don't just add all the functions in the class instead of some here and some below
# Helper functions
def normal(theta):
    """Compute cos and sin of an angle

    :param theta: angle
    :return: cos and sin
    """
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = jit(vmap(normal))

def multiply_masks(mask1, mask2):
    """Multiply two masks of 0s and 1s

    :param mask1: mask1
    :param mask2: mask2
    :return: final mask
    """
    return mask1 * mask2

def distance(point1, point2):
    """Compute the distance between two points

    :param point1: coordinates of point 1
    :param point2: coordinates of point 2
    :return: distance
    """
    diff = point1 - point2
    squared_diff = jnp.sum(jnp.square(diff))
    return jnp.sqrt(squared_diff)

distance = jit(vmap(distance, in_axes=(None, 0)))

# Change the angle and speed of the agent a bit 
def move(obs, key):
    """(Randomly) Change the velocity vector of agents to make them move

    :param obs: agent observation (unused at the moment)
    :param key: jax PRNGKey
    :return: new velocity vector
    """
    return random.normal(key, shape=(3,)) * MOVE_SCALE

move = jit(vmap(move, in_axes=(0, 0)))

def eat(agent_idx, state):
    """Make the agent eat food around itself

    :param agent_idx: idx of the agent in the state
    :param state: current state
    :return: updated food_exist mask, number of eaten food
    """
    agent_pos = state.agents.pos[agent_idx]
    alive = state.agents.alive[agent_idx]

    food_position = state.objects.pos
    food_exist = state.objects.exist

    food_dist = distance(agent_pos, food_position)
    can_eat_idx = jnp.where(food_dist < FISH_EATING_RANGE, 1, 0)
    can_eat_idx = jnp.where(can_eat_idx, food_exist, 0)

    n_eaten = jnp.sum(can_eat_idx)

    mask = jnp.where(can_eat_idx, 0, 1)
    updated_food_exist = food_exist * mask

    # Return the old mask if the agent isn't alive
    food_exist = jnp.where(alive, updated_food_exist, food_exist)
    n_eaten = jnp.where(alive, n_eaten, 0)
    return food_exist, n_eaten 

eat = vmap(eat, in_axes=(0, None))

    
class Aquarium(BaseEnv):
    """ Minimalistic aquarium environmnent"""
    def __init__(self, max_agents=10, max_objects=50, grid_size=50):
        self.max_agents = max_agents
        self.max_objects = max_objects  
        self.grid_size = grid_size
        # Drop food when only 10% of max food remaining in the aquarium
        self.drop_food_threshold = self.max_objects // 10 
        # Add back half of the max food in the aquarium when condition above is met
        self.n_dropped_food = self.max_objects // 2 

    def init_state(self, num_agents=None, num_obs=2, seed=0):
        num_agents = num_agents if num_agents else self.max_agents
        key = random.PRNGKey(seed)
        agents_key_pos, agents_key_vel, agents_color_key, objects_key_pos = random.split(key, 4)
        fish_velocity = random.uniform(agents_key_vel, shape=(self.max_agents, N_DIMS), minval=-1, maxval=1)
        fish_velocity = (fish_velocity / jnp.linalg.norm(fish_velocity)) * FISH_SPEED
        # Don't add fish too close to the surface / border of the aquarium
        min_border_distance = self.grid_size // 10

        fish = Agents(
            pos=random.uniform(key=agents_key_pos, shape=(self.max_agents, N_DIMS), minval=min_border_distance, maxval=self.grid_size - min_border_distance),
            velocity=fish_velocity,
            alive=jnp.hstack((jnp.ones(num_agents), jnp.zeros(self.max_agents - num_agents))),
            color=jnp.full((self.max_agents, 3), jnp.array(FISH_COLOR)),
            obs=jnp.zeros((self.max_agents, num_obs)),
            energy=jnp.zeros((self.max_agents,))
        )
        
        # Add food at the surface of the aquarium
        food_pos = self.place_food_at_surface(self.max_objects, objects_key_pos)
        # Only add half of existing food at the beginning
        half = self.max_objects // 2
        exists_food = jnp.hstack((jnp.ones(half), jnp.zeros(half)))
        
        food = Objects(
            pos=food_pos,
            velocity=jnp.tile(jnp.array([0., 0., -1]), (self.max_objects, 1)) * FOOD_SPEED,
            color=jnp.full((self.max_objects, 3), jnp.array(FOOD_COLOR)),
            exist=exists_food
        )

        aquarium_state =  AquiariumState(
            time=0,
            grid_size=self.grid_size,
            agents=fish,
            objects=food
        )
        
        return aquarium_state
    
    # TODO : add key in the state to remove it from those arguments
    # TODO : split the code into different functions to make it more digestible
    @partial(jit, static_argnums=(0,))
    def _step(self, state, key):
        """Make a jitted step in the environment

        :param state: old state
        :param key: jax PRNGKey
        :return: new state
        """
        # Drop new food if there is not enough in the aquarium, else keep the same state 
        key, drop_food_key = random.split(key)
        drop_food = jnp.sum(state.objects.exist) < self.drop_food_threshold 
        non_ex_food_idx = jnp.where(state.objects.exist == 0, 1., 0.)
        state = jax.lax.cond(drop_food, self.drop_food, self.dont_change, state, non_ex_food_idx, drop_food_key)

        # Update agents positions
        keys = random.split(key, self.max_agents)
        d_vel = move(state.agents.obs, keys)
        velocity = state.agents.velocity + d_vel
        velocity = (velocity / jnp.linalg.norm(velocity)) * FISH_SPEED
        agents_pos = state.agents.pos + velocity

        # Collide with walls
        agents_pos = jnp.clip(agents_pos, 0, self.grid_size)

        # Update food position
        cur_food_pos = state.objects.pos
        exist_food = jnp.where(state.objects.exist != 0.0, 1, 0)
        # Adapt mask to pos shape
        exist_food = jnp.broadcast_to(jnp.expand_dims(exist_food, 1), cur_food_pos.shape)
        food_vel = state.objects.velocity
        new_food_pos = cur_food_pos + food_vel
        food_pos = jnp.where(exist_food, new_food_pos, cur_food_pos)
        food_pos = jnp.clip(food_pos, 0, self.grid_size)

        # Compute the exist food idx after each agent has eaten nearby food
        agent_idx = jnp.arange(0, self.max_agents)
        food_exist, eaten = eat(agent_idx, state)
        agents_energy = state.agents.energy + eaten

        # Multiply the masks between them to get the final exist array
        food_exist = reduce(multiply_masks, food_exist)

        # Update new state
        time = state.time + 1
        objects = state.objects.replace(pos=food_pos, exist=food_exist)
        agents = state.agents.replace(pos=agents_pos, velocity=velocity, energy=agents_energy)
        state = state.replace(time=time, agents=agents, objects=objects)
        return state
    
    def step(self, state, key):
        """Make a step in the environment by calling _step

        :param state: old_state
        :param key: jax PRNGKey
        :return: new state
        """
        # At the moment only call the _step but could add new methods in the future
        state = self._step(state, key)
        return state
    
    # function to modify the state in the lax.cond for food dropping
    def drop_food(self, state, non_ex_food_idx, key):
        """Place unexisting food at the top of the aquarium to make it drop

        :param state: current state
        :param non_ex_food_idx: mask of non existing food
        :param key: jax PRNGKey
        :return: updated state with new food placed at the top of the water
        """
        # Get the idx of the new food that will become existing
        idx_key, pos_key = random.split(key)
        idx = random.choice(idx_key, a=jnp.arange(len(non_ex_food_idx)), p=non_ex_food_idx, shape=(self.n_dropped_food,), replace=False)
        # Assign new positions and existing values to the food objects at these idx
        food_exists = state.objects.exist.at[idx].set(1.)
        new_pos = self.place_food_at_surface(self.n_dropped_food, pos_key)
        food_pos = state.objects.pos.at[idx].set(new_pos)
        objects = state.objects.replace(pos=food_pos, exist=food_exists)
        return state.replace(objects=objects)
    
    # function to keep the state as it is in the lax.cond for food dropping
    def dont_change(self, state, non_ex_food_idx, key):
        """Keep the state intact

        :param state: current state
        :param non_ex_food_idx: mask of non existing food
        :param key: jax PRNGKey
        :return: unmodified state
        """
        return state

    def place_food_at_surface(self, n_food, key):
        """Place n_food at the surface of the aquarium

        :param n_food: number of food particles
        :param key: _description_jax PRNGKey
        :return: food position
        """
        x_y_food_pos=random.uniform(key=key, shape=(n_food, 2), minval=0, maxval=self.grid_size) 
        z_food_pos = jnp.full((n_food, 1), fill_value=self.grid_size)
        food_pos = jnp.concatenate((x_y_food_pos, z_food_pos), axis=1)
        return food_pos

    def add_agent(self, state, agent_idx):
        """Add an agent at idx agent_idx

        :param state: state
        :param agent_idx: agent_idx
        :return: state with agent agent_idx alive
        """
        agents = state.agents.replace(alive=state.agents.alive.at[agent_idx].set(1.0))
        state = state.replace(agents=agents)
        return state
    
    def remove_agent(self, state, agent_idx):
        """Remove an agent at idx agent_idx

        :param state: state
        :param agent_idx: agent_idx
        :return: state with agent agent_idx dead
        """
        agents = state.agents.replace(alive=state.agents.alive.at[agent_idx].set(0.0))
        state = state.replace(agents=agents)
        return state

    # TODO : See how to potentially render the env with PIL / VideoWriter like in other jax libs
    @staticmethod
    def render(state):
        """Render the current state 

        :param state: state
        """
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
        agents_energy = state.agents.energy[alive_agents]

        exist_object = jnp.where(state.objects.exist != 0.0)
        objects_x_pos = state.objects.pos[:, 0][exist_object]
        objects_y_pos = state.objects.pos[:, 1][exist_object]
        objects_z_pos = state.objects.pos[:, 2][exist_object]
        objects_colors = state.objects.color[exist_object]

        SCALE = 15
        # TODO : see how to add cmap=colormaps["gist_rainbow"] for fish colors
        ax.scatter(agents_x_pos, agents_y_pos, agents_z_pos, c=agents_colors, s=(1 +agents_energy)*SCALE, marker="o", label="Fish")
        ax.scatter(objects_x_pos, objects_y_pos, objects_z_pos, c=objects_colors, marker="o", label="Food")

        ax.set_title("Aquarium Simulation")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_xlim(0, state.grid_size)
        ax.set_ylim(0, state.grid_size)
        ax.set_zlim(0, state.grid_size)

        ax.legend()

        plt.draw()
        plt.pause(0.001)
