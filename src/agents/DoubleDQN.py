import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, losses

from agents.RL_algorithm import RLAlgorithm

class DoubleDQNagent(RLAlgorithm):
    '''
    Class that represents the agent that will interact with the environment
    '''

    def __init__(self, state_size, action_size, hyperparams):
        """
        Initializes the DoubleDQNagent with specified state size, action size, and hyperparameters.

        :param state_size: Size of the state space.
        :type state_size: int
        :param action_size: Size of the action space.
        :type action_size: int
        :param hyperparams: Hyperparameters for the agent.
        :type hyperparams: dict
        """
        super().__init__(state_size, action_size, hyperparams)
        self.policyNN = self.build_model()
        self.targetNN = self.build_model()
        self.targetNN.set_weights(self.policyNN.get_weights())

    def build_model(self):
        """
        Builds the neural network model for the agent.

        :return: A compiled Keras model.
        :rtype: tf.keras.Model
        """
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=losses.mse, optimizer=optimizers.legacy.Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def learn_replay(self, batch_size):
        """
        Trains the agent using the experience replay technique.

        :param batch_size: The size of the minibatch to sample from the replay buffer.
        :type batch_size: int
        :return: The loss value from the training step.
        :rtype: float
        """
        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.buffer, batch_size)

        # Reshape states to ensure they have the correct shape
        states = np.array([np.reshape(sample[0], [1, self.state_size]) for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([np.reshape(sample[3], [1, self.state_size]) for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Reshape states and next_states to match batch size
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        # Double DQN
        # Get the actions that policyNN predicts for the next states
        policy_actions = np.argmax(self.policyNN.predict(next_states), axis=1)

        # Evaluate those actions with the targetNN
        sample_qhat_next = self.targetNN.predict(next_states)
        dones = np.expand_dims(dones, axis=1)
        sample_qhat_next *= (1 - dones)
        sample_qhat_next = sample_qhat_next[np.arange(batch_size), policy_actions]

        target = self.policyNN.predict(states)

        # Update target values using advanced indexing
        target[np.arange(batch_size), actions.flatten()] = rewards + self.gamma * sample_qhat_next

        history = self.policyNN.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        return loss

    def save_model(self, name):
        """
        Saves the model to a file.

        :param name: The name of the file to save the model.
        :type name: str
        """
        self.policyNN.save(name)
        self.targetNN.save(name.split('.')[0] + '_target.h5')

    def load_model(self, name):
        """
        Loads the model from a .h5 file.

        :param name: The name of the file to load the model from.
        :type name: str
        """
        self.policyNN.load_weights(name)
        self.targetNN.load_weights(name.split('.')[0] + '_target.h5')

    def update_target_network(self):
        """
        Updates the target network with the policy network weights.
        """
        self.targetNN.set_weights(self.policyNN.get_weights())
