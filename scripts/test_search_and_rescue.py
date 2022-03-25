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
from mae_envs.envs.search_and_rescue import make_env


core_dir = abspath(join(dirname(__file__), ".."))
envs_dir = "mae_envs/envs"
xmls_dir = "xmls"

if __name__ == "__main__":
    """
    Actions:
    - action_movement: [n_agents, 3] (x, y, z forces)
    - action_pull: [n_agents]
    - action_glueall: [n_agents]
    """
    agent_types = [
        {
            "is_rescuer": True,
            "view_range": 5,
            "has_lidar": True,
            "lidar_range": 6,
            "model_id": 0,
            "type": "rescuer",
        },
        {
            "is_rescuer": False,
            "view_range": 10,
            "has_lidar": True,
            "lidar_range": 6,
            "model_id": 1,
            "type": "seeker",
        },
        {
            "is_rescuer": False,
            "view_range": 0,
            "has_lidar": False,
            "lidar_range": 0,
            "model_id": -1,
            "type": "hiker",
        },
    ]
    env = make_env(agent_types=agent_types)
    done = False
    obs = env.reset()
    # obs dim (n_agents, )
    while not done:
        action = np.random.randint(0, 11, size=(env.n_agents, 3))
        obs, reward, done, info = env.step(action)
        env.render()
        print(action)
