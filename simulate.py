import time 
import argparse

from jax import random

from simulationsandbox.environments.two_d_example_env import TwoDEnv
from simulationsandbox.environments.three_d_example_env import ThreeDEnv
from simulationsandbox.utils.sim_types import SIMULATIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--max_agents", type=int, default=10)
    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=60)
    parser.add_argument("--step_delay", type=float, default=0.01)
    parser.add_argument("--sim_type", type=str, default="two_d")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_false")
    args = parser.parse_args()  

    key = random.PRNGKey(args.random_seed)

    try:
        Simulation = SIMULATIONS[args.sim_type]
    except:
        raise(ValueError(f"Unknown sim type {args.sim_type}"))

    sim = Simulation(args.max_agents, args.grid_size)
    state = sim.init_state(args.num_agents, args.num_obs, key)

    # Launch a simulation
    print("Simulation started")
    for timestep in range(args.num_steps):
        time.sleep(args.step_delay)
        key, a_key, step_key = random.split(key, 3)

        if timestep % 10 == 0:
            print(f"\nstep {timestep}")

        if timestep == (args.num_steps // 3):
            # Add 3 agents and change the color of an agent
            state = sim.add_agent(state, 7)
            state = sim.add_agent(state, 9)
            state = sim.add_agent(state, 5)
            state = state.replace(colors=state.colors.at[0, 2].set(1.0))

        if timestep ==  2* (args.num_steps // 3):
            # Remove 3 other agents and change the color of another agent
            state = sim.remove_agent(state, 2)
            state = sim.remove_agent(state, 1)
            state = sim.remove_agent(state, 4)
            state = state.replace(colors=state.colors.at[7, 1].set(1.0))


        actions = sim.choose_action(state.obs, a_key)
        state = sim.step(state, actions, step_key)

        if args.visualize:
            Simulation.visualize_sim(state)
    print("\nSimulation ended")

if __name__ == "__main__":
    main()
