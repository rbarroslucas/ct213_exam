import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, losses

from agents.RL_algorithm import RLAlgorithm

class DuelDQNagent(RLAlgorithm):
    '''
    Class that represents the agent that will interact with the environment
    '''

    def __init__(self, state_size, action_size, hyperparams):
        '''
        Initializes the DuelDQNagent.

        :param state_size: The size of the state space.
        :type state_size: int
        :param action_size: The size of the action space.
        :type action_size: int
        :param hyperparams: Hyperparameters for the agent.
        :type hyperparams: dict
        '''
        super().__init__(state_size, action_size, hyperparams)
        self.policyNN = self.build_model()

    def build_model(self):
        '''
        Builds the neural network model for the agent.

        :return: The compiled neural network model.
        :rtype: tf.keras.Model
        '''
        inputs = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(256, activation='relu')(inputs)
        dense2 = layers.Dense(128, activation='relu')(dense1)

        # State value stream
        state_value = layers.Dense(1, activation='linear')(dense2)

        # Advantage stream
        advantages = layers.Dense(self.action_size, activation='linear')(dense2)

        # Combine state value and advantages to get Q-values
        outputs = state_value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses.MeanSquaredError(),
                      optimizer=optimizers.legacy.Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def learn_replay(self, batch_size):
        '''
        Trains the agent using the experience replay technique.

        :param batch_size: The size of the minibatch to sample from the replay buffer.
        :type batch_size: int
        :return: The loss value from the training step.
        :rtype: float
        '''
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

        # Predict Q-values for current states and next states
        q_values = self.policyNN.predict(states, verbose=0)
        next_q_values = self.policyNN.predict(next_states, verbose=0)

        # Update Q-values for the actions taken
        for i in range(batch_size):
            if not dones[i]:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            else:
                q_values[i][actions[i]] = rewards[i]

        # Train the model on the updated Q-values
        history = self.policyNN.fit(states, q_values, epochs=1, verbose=0)

        # Return the loss value from the training
        loss = history.history['loss'][0]
        return loss

    def save_model(self, name):
        '''
        Saves the model to a file.

        :param name: The name of the file to save the model to.
        :type name: str
        '''
        self.policyNN.save(name)

    def load_model(self, name):
        '''
        Loads the model from a file.

        :param name: The name of the file to load the model from.
        :type name: str
        '''
        self.policyNN.load_weights(name)

    def update_target_network(self):
        '''
        Just to mantain the same interface as the other agents, but it is not necessary for DuelDQN
        '''
        pass
