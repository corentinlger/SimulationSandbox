import time 
import argparse

from jax import random

from simulationsandbox.utils.envs import ENVS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="aquarium")
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--max_agents", type=int, default=10)
    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--step_delay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_false")
    args = parser.parse_args()  

    key = random.PRNGKey(args.seed)

    try:
        Env = ENVS[args.env]
    except:
        raise(ValueError(f"Unknown environment {args.env}"))

    env = Env(max_agents=args.max_agents, grid_size=args.grid_size)
    state = env.init_state(seed=args.seed)

    # Launch a simulation
    print("Simulation started")
    for timestep in range(args.num_steps):
        time.sleep(args.step_delay)
        key, step_key = random.split(key)

        if timestep % 10 == 0:
            print(f"\nstep {timestep}")

        state = env.step(state, step_key)

        if args.visualize:
            Env.render(state)

    print("\nSimulation ended")

if __name__ == "__main__":
    main()
