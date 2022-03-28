import collections
import numpy as np





'''
    Dummy node class representing the non-existent parent node of tree root node.
'''
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


'''

'''
class SearchNode(object):
    def __init__(self, state, action=None, parent=DummyNode()):
        self.state = state
        self.action = action
        self.is_expanded = False
        self.is_terminal = False
        self.parent = parent

        self.n_visits = 0
        self.total_value = 0.0

        self.children = {}
        self.child_priors = np.zeros(, dtype=np.float32)
        self.child_total_values = np.zeros(, dtype=np.float32)
        self.child_number_visits = np.zeros(, dtype=np.int32)


    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)


    def child_U(self):
        return self.c_puct * self.child_priors * np.sqrt(np.log(self.n_visits + 1) / (1 + self.child_number_visits))

    def best_child(self):
        if self.state.terminal:
            return None, None, None

        q = self.child_Q()
        u = self.child_U()
        child_values = q + u

        legal_actions = self.state.legal_actions
        illegal_actions = np.ones(child_values.shape, dtype=bool)
        illegal_actions[legal_actions] = False
        child_values[illegal_actions] = -np.inf

        best = np.random.choice(np.flatnonzero(np.isclose(child_values, child_values.max())))

        return best, child_values[best], child_values


    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors


    def add_child(self, action):
        successor = self.state.step(action)
        self.children[action] = UCTNode(successor, action, parent=self)


'''
'''
class SearchProblem():
    def __init__(self, root, params):
        self.root = root
        self.params = params

    def rollout(self, node=None):
        node = self.root
        n_rollout = 0
        while node.is_expanded && n_rollout < self.params.max_rollouts:
            node.n_visits += 1
            if node.parent is not None:
                node.parent.child_number_visits[node.action] += 1
            action, value, values = node.best_child()

            if actions is None:
                node.is_terminal = True
                break

            if action not in node.children:
                node.add_child(action)

            node = node.children[action]

        return node

    def backup(self, node, value_estimate):
        node.n_visits += 1
        node.total_value += value_estimate

        if node.parent is not None:
            node.parent.child_number_visits[node.action] += 1
            node.parent.child_total_value[node.action] += value_estimate
            node = node.parent

        while node.parent is not None:
            node.total_value += value_estimate
            node.parent.child_total_value[node.action] += value_estimate
            node = node.parent