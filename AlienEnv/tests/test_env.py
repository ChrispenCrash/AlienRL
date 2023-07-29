from stable_baselines3.common.env_checker import check_env
from alienrl_env import AlienRLEnv

env = AlienRLEnv()

try:
    check_env( env, warn=True)
except AssertionError as msg:
    print(msg)

env.close()