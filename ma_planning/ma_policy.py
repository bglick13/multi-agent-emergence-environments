import numpy as np

from collections import deque
from typing import Union
from torch import nn, FloatTensor, LongTensor
from torch.functional import F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from mae_envs.envs import DraftState
from mcts import SearchNode, SearchProblem



class SwarmAgent():
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.macro_action = None

    def set_action(self, action):
        self.macro_action = action

    def act(self):
        return self.macro_action


class CaptainAgent():
    def __init__(self, model, env, agents):
        self.model = model
        self.best_model = model
        self.env = env
        self.agents = agents
        self.solver = None

    def simulate(self):
        leaf = self.solver.rollout()
        value = self.evaluate_leaf(leaf)
        self.solver.backup(leaf, value)
        return leaf

    def get_action(self, obs, num_reads=100, action=-1, random=False):
        if self.solver is None:
            self.root = SearchNode(obs, action)
            self.solver = SearchProblem(self.root)
        else:
            self.root = SearchNode(obs, action, self.root)
            self.solver.root = self.root

        leafs = []
        for _ in range(num_reads):
            leafs.append(self.simulate())

        action, value, values = self.root.best_child()
        successor, _, _, _ = env.step(action)
        nn_probs, nn_value = self.get_preds(successor)
        p = F.softmax(FloatTensor(values), -1).numpy()
        if random:
            action = np.random.choice(range(len(values)), p=p)
        else:
            top5 = values.argsort()[-5:]
            _p = F.softmax(FloatTensor(values[top5]), -1).numpy()
            action = np.random.choice(top5, p=_p)
        return action, values, p, nn_value, leafs

    def get_preds(self, obs):
        s_in = torch.FloatTensor(obs)
        s_in.requires_grad = False
        encoded_s = self.model.forward(s_in)
        probs = self.model.get_next_action_output(encoded_s) # n_agents x 3 x 11
        probs = F.softmax(torch.FloatTensor(probs)).detach().cpu().numpy()
        value = F.softmax(self.model.get_value_output(encoded_s)).detach().cpu().numpy()
        return probs, value

    def evaluate_leaf(self, leaf):
        probs, value = self.get_preds(leaf)
        if not leaf.is_terminal:
            leaf.expand(probs)
        return value