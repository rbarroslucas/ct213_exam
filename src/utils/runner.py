import os
import csv
import gym
import time
import numpy as np

from agents.DQN import DQNagent
from agents.DoubleDQN import DoubleDQNagent
from agents.DuelDQN import DuelDQNagent
from utils.utils import FileHandler, mid_plots, create_folder


class Game_Runner:
    def __init__(self, algorithm, minibatch_size, hyperparams, id='LunarLander-v2', render_mode=None, seed=None):
        """
        Initializes the Game Runner with specified algorithm, minibatch size, hyperparameters, environment ID, render mode, and seed.

        :param algorithm: The algorithm to use (DQN, DoubleDQN, DuelDQN).
        :type algorithm: str
        :param minibatch_size: The size of minibatches for experience replay.
        :type minibatch_size: int
        :param hyperparams: Hyperparameters for the agent.
        :type hyperparams: tuple
        :param id: Environment ID for Gym.
        :type id: str, optional
        :param render_mode: Mode to render the environment.
        :type render_mode: str, optional
        :param seed: Seed for the environment.
        :type seed: int, optional
        """
        self.algorithm = algorithm
        self.minibatch = minibatch_size
        self.hyperparams = hyperparams
        self.env = self.create_env(id, render_mode, seed)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = self.create_agent()
        self.folder = create_folder(self.algorithm)
        self.FileManager = FileHandler(self.folder, self.algorithm)

    def create_env(self, id, render_mode, seed):
        """
        Creates the environment using Gym.

        :param id: Environment ID for Gym.
        :type id: str
        :param render_mode: Mode to render the environment.
        :type render_mode: str, optional
        :param seed: Seed for the environment.
        :type seed: int, optional
        :return: The created Gym environment.
        :rtype: gym.Env
        """
        return gym.make(id=id, render_mode=render_mode, seed=seed) if seed else gym.make(id=id, render_mode=render_mode)

    def create_agent(self):
        """
        Creates the agent based on the specified algorithm.

        :return: The created agent.
        :rtype: RLAlgorithm
        """
        if self.algorithm == 'DQN':
            return DQNagent(self.state_size, self.action_size, self.hyperparams)
        elif self.algorithm == 'DoubleDQN':
            return DoubleDQNagent(self.state_size, self.action_size, self.hyperparams)
        elif self.algorithm == 'DuelDQN':
            return DuelDQNagent(self.state_size, self.action_size, self.hyperparams)
        else:
            raise ValueError("Unsupported algorithm: {}".format(self.algorithm))

    def train(self, train_episodes, nn_name, render=False, fig_format='png'):
        """
        Trains the agent in the environment for a specified number of episodes.

        :param train_episodes: Number of episodes to train.
        :type train_episodes: int
        :param nn_name: Name of the neural network file to save.
        :type nn_name: str
        :param render: Whether to render the environment during training.
        :type render: bool, optional
        :param fig_format: Format for saving plots.
        :type fig_format: str, optional
        """
        start_time = time.time()
        loss_history = []
        return_history = []

        for episode in range(1, train_episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            score = 0.0
            for steps in range(1, 1001):
                if render:
                    self.env.render()
                action = self.agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                score += reward
                state = next_state
                self.agent.steps += 1
                if done:
                    print("episode: {}/{}, steps: {}, score: {:.6f}, epsilon: {:.3f}".format(episode, train_episodes,
                                                                                             steps, score,
                                                                                             self.agent.epsilon))
                    break
                if len(self.agent.buffer) > self.minibatch:
                    loss = self.agent.learn_replay(self.minibatch)
                    loss_history.append(loss)
                if self.agent.steps % 100 == 0:
                    self.agent.update_target_network()
            return_history.append(score)
            self.agent.update_epsilon()
            if episode % 20 == 0:
                self.save(nn_name)
                self.FileManager.save_return_history(return_history)
                mid_plots(self.algorithm, return_history, loss_history, self.agent, self.folder, fig_format)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.env.close()
        self.FileManager.save_training_summary(train_episodes, self.hyperparams, self.minibatch, elapsed_time,
                                               self.agent.steps)

    def evaluate(self, test_episodes, load_agent, nn_name='lunar_lander_dqn.h5', render=False, fig_format='png'):
        """
        Evaluates the trained agent in the environment for a specified number of episodes.

        :param test_episodes: Number of episodes to evaluate.
        :type test_episodes: int
        :param load_agent: Whether to load the agent's weights from a file.
        :type load_agent: bool, optional
        :param nn_name: Name of the neural network file to load.
        :type nn_name: str, optional
        :param render: Whether to render the environment during evaluation.
        :type render: bool, optional
        :param fig_format: Format for saving plots.
        :type fig_format: str, optional
        """
        if load_agent:
            self.load_weights(nn_name)
        return_history = []

        file_path = os.path.join(self.folder, 'evaluate.csv')
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "time", "score"])

        for episode in range(1, test_episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            score = 0.0
            for step in range(1, 1001):
                if render:
                    self.env.render()
                action = self.agent.get_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                score += reward
                if done:
                    print("episode: {}/{}, time: {}, score: {:.6}"
                          .format(episode, test_episodes, step, score))
                    with open(file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([episode, step, score])
                    break
            return_history.append(score)
        print(f'Testing completed. Saved results in {file_path}')

    def load_weights(self, name):
        """
        Loads the agent's weights from a file.

        :param name: Name of the neural network file to load.
        :type name: str
        """
        if os.path.exists(name):
            print('Loading weights from previous learning session.')
            self.agent.load_model(name)
        else:
            print('No weights found from previous learning session.')
            exit(-1)

    def save(self, name):
        """
        Saves the agent's weights to a file.

        :param name: Name of the neural network file to save.
        :type name: str
        """
        self.agent.save_model(os.path.join(self.folder, name))