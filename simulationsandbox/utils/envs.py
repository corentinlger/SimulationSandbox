from simulationsandbox.environments.two_d_example_env import TwoDEnv
from simulationsandbox.environments.three_d_example_env import ThreeDEnv
from simulationsandbox.environments.lake_env import LakeEnv
from simulationsandbox.environments.aquarium import Aquarium

ENVS = {"two_d": TwoDEnv,
        "three_d": ThreeDEnv,
        "lake": LakeEnv,
        "aquarium": Aquarium
        }
