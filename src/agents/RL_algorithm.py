import numpy as np
from collections import deque


class RLAlgorithm():
    '''
    Class that represents the Double Dueling Deep Q-Learning agent
    '''

    def __init__(self,
                 state_size,
                 action_size,
                 hyperparams):

        self.state_size = state_size
        self.action_size = action_size

        self.steps = 0

        gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, buffer_size = hyperparams

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.buffer = deque(maxlen=buffer_size)
        self.policyNN = None
        self.targetNN = None

    def remember(self, state, action, reward, next_state, done):
        '''
        Adds the state, action, reward, next_state and done tuple to the buffer
        :param state: the current state of the environment
        :param action: the action taken by the agent
        :param reward: the reward received by the agent
        :param next_state: the next state of the environment
        :param done: whether the episode is done or not
        '''
        self.buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        '''
        Updates the epsilon value of the agent
        '''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action(self, state, training):
        '''
        Returns the action to be taken by the agent following the epsilon-greedy policy
        :param state: the current state of the environment
        :return: the action to be taken
        '''
        if training:
            return self.get_eps_greedy_action(state)
        else:
            return self.get_greedy_action(state)

    def get_eps_greedy_action(self, state):
        '''
        Returns the action to be taken by the agent following the epsilon-greedy policy
        :param state: the current state of the environment
        :return: the action to be taken
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        '''
        Returns the action to be taken by the agent following the greedy policy
        :param state: the current state of the environment
        :return: the action to be taken
        '''
        state = np.reshape(state, [1, self.state_size])
        return np.argmax(self.policyNN.predict(state, verbose=0))

    def build_model(self):
        '''
        Builds the dueling neural network model for the agent
        '''
        raise NotImplementedError('Please implement this method')

    def learn_replay(self, batch_size):
        '''
        Trains the agent using the experience replay technique
        '''
        raise NotImplementedError('Please implement this method')

    def save_model(self, name='dddqn_model.h5'):
        '''
        Saves the model to a file
        '''
        raise NotImplementedError('Please implement this method')

    def load_model(self, name='dddqn_model.h5'):
        '''
        Loads the model from a file
        '''
        raise NotImplementedError('Please implement this method')

    def update_target_network(self):
        raise NotImplementedError('Please implement this method')