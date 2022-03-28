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





class CaptainAgent():
    def __init__(self, model, env):
        self.model = model
        self.best_model = model
        self.env = env
        self.solver = None

    def simulate(self):
        leaf = self.solver.rollout()
        value = self.evaluate_leaf(leaf)
        self.solver.backup(leaf, value)
        return leaf

    def get_action(self, state, num_reads=100, action=-1, deterministic=False):
        if self.solver is None:
            self.root = SearchNode(state, action)
            self.solver = SearchProblem(self.root)
        else:
            self.root = SearchNode(state, action, self.root)
            self.solver.root = self.root

        leafs = []
        for _ in range(num_reads):
            leafs.append(self.simulate())
        action, value, values = self.root.best_child()

        successor, _, _, _ = env.step(action)

        #TODO: fill this in once nn is implemented
        nn_probs, nn_value, _ = self.get_preds(successor, plot_attn=True)

        p = F.softmax(FloatTensor(values), -1).numpy()
        if not deterministic:
            action = np.random.choice(range(len(values)), p=p)
        else:
            top5 = values.argsort()[-5:]
            _p = F.softmax(FloatTensor(values[top5]), -1).numpy()
            action = np.random.choice(top5, p=_p)
        return action, values, p, nn_value, leafs

    def get_preds(self):
        pass

    def evaluate_leaf(self, leaf):
        probs, value, legal_moves = self.get_preds(leaf)
        if not leaf.is_terminal:
            leaf.expand(probs)
        return value