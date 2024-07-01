import numpy as np

class StateDiscretizer:
    def __init__(self, bins):
        self.bins = bins
    
    def discretize(self, state):
        discretized_state = []
        for i, space in enumerate(self.bins):
            if space is not None:
                discretized_state.append(np.digitize(state[i], space))
            else:
                discretized_state.append(state[i])
        return tuple(discretized_state)

