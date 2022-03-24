import logging
import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments
from mae_envs.envs.hide_and_seek import make_env


core_dir = abspath(join(dirname(__file__), ".."))
envs_dir = "mae_envs/envs"
xmls_dir = "xmls"

if __name__ == "__main__":
    env = make_env()
    done = False
    obs = env.reset()
    # obs dim (n_agents, )
    while not done:
        action = np.random.randint(0, env.action_space.n)
        obs, reward, done, info = env.step(action)
        env.render()
