import os
import csv
import gym
import numpy as np
import matplotlib.pyplot as plt

from utils.params import *
from agents.DQN import DQNagent
from agents.DoubleDQN import DoubleDQNagent
from agents.DuelDQN import DuelDQNagent

'''
This script has the only purpose of evaluating the agent's performance on the Lunar Lander environment. 
It is used to evaluate weights from previous learning sessions, when the user is not interested in training the agent.
'''

# Initiating the Lunar Lander environment
env = gym.make(
    "LunarLander-v2",
    render_mode=RENDER_MODE
)

# Setting the environment parameters and agent hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialising the agent
if ALGORITHM == 'DQN':
    agent = DQNagent(state_size, action_size, HYPERPARAMS)
elif ALGORITHM == 'DoubleDQN':
    agent = DoubleDQNagent(state_size, action_size, HYPERPARAMS)
elif ALGORITHM == 'DuelDQN':
    agent = DuelDQNagent(state_size, action_size, HYPERPARAMS)

# Sample Size and history
return_history = []
loss_history = []
frames = 0

# Checking if weights from previous learning session exists
if os.path.exists(NN_NAME):
    print('Loading weights from previous learning session.')
    agent.load_model(NN_NAME)
else:
    print('No weights found from previous learning session.')
    exit(-1)

with open('evaluate.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["episode", "time", "score"])

for episode in range(1, TEST_EPISODES + 1):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score, steps, done = 0.0, 0, False
    while not done:
        if RENDER:
            env.render()

        action= agent.get_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        score += reward

        state = next_state
        steps += 1

        if done:
            print("episode: {}/{}, time: {}, score: {:.6}".format(episode, TEST_EPISODES, steps, score))
            with open('evaluate.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, steps, score])
            break
    return_history.append(score)
    frames += steps
# Prints mean return
print('Mean return: ', np.mean(return_history))

# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.' + fig_format, format=fig_format)

