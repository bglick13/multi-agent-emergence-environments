import gym
import numpy as np
from mae_envs.wrappers.util import update_obs_space


class AgentType(gym.ObservationWrapper):
    """
    This wrapper just stores team membership information at initialization.
    The information is stored as a key in the self.metadata property, which ensures
    that it is available even if this wrapper is not on top of the wrapper
    hierarchy.

    Arguments:
        team_index: list/numpy vector of team membership index
                    length must be equal to number of agents
                    e.g. [0,0,0,1,1,1] means first 3 agents are in team 0,
                    second 3 agents in team 1
        n_teams: if team_index is None, agents are split in n_teams number
                 of teams, with as equal team sizes as possible.
                 if team_index is set, this argument is ignored

    One planned use of this wrapper is to evaluate the "TrueSkill" score
    during training, which requires knowing which agent belongs to which team

    Note: This wrapper currently does not align the reward structure with the
          teams, but that could be easily implemented if desired.
    """

    def __init__(self, env, agent_specs):
        super().__init__(env)
        self.n_agents = self.metadata["n_actors"]

        # store in metadata property that gets automatically inherited
        # make sure we copy value of team_index if it's a numpy array
        self.metadata["agent_types"] = agent_specs
        self.observation_space = update_obs_space(
            env, {"team_size": (self.n_agents, 1)}
        )
        self.observation_space = update_obs_space(env, {"model_id": (self.n_agents, 1)})

    def observation(self, obs):
        obs["model_type"] = np.array(
            [self.metadata["agent_types"][i]["model_id"] for i in range(self.n_agents)]
        )
        return obs
