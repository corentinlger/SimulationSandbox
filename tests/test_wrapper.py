import time

from simulationsandbox.utils.envs import ENVS
from simulationsandbox.simulator_wrapper import SimulationWrapper

MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
SLEEP_TIME = 2
STEP_DELAY = 0.01
PRINT_DATA = False
SEED = 0

def test_wrapper():
    for Env in ENVS.values():
        env = Env(max_agents=MAX_AGENTS, grid_size=GRID_SIZE)
        state = env.init_state(seed=SEED)

        assert env.max_agents == MAX_AGENTS
        assert env.grid_size == GRID_SIZE
        assert state.agents.pos.shape[0] == MAX_AGENTS

        sim = SimulationWrapper(env, state, seed=SEED, step_delay=STEP_DELAY, print_data=PRINT_DATA)

        print('Started')
        sim.start()
        time.sleep(SLEEP_TIME)
        assert sim.running is True

        sim.pause()
        print('Paused')
        time.sleep(SLEEP_TIME)
        assert sim.paused is True

        print('Resumed')
        sim.resume()
        time.sleep(SLEEP_TIME)
        assert sim.running is True

        sim.stop()
        print('stopped')
        assert sim.stop_requested is True

