from simulationsandbox.utils.envs import ENVS

MAX_AGENTS = 10
GRID_SIZE = 20 
SEED = 0

def test_simulation_init():
    for Env in ENVS.values():
        env = Env(max_agents=MAX_AGENTS, grid_size=GRID_SIZE)
        state = env.init_state(seed=SEED)

        assert env.max_agents == MAX_AGENTS
        assert env.grid_size == GRID_SIZE
        assert state.agents.pos.shape[0] == MAX_AGENTS

