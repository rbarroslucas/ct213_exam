import numpy as np
import gym
import matplotlib.pyplot as plt

from utils import StateDiscretizer
from reinforcement_learning import Sarsa, QLearning

NUM_EPISODES = 400  # Number of episodes used for training
RENDER = False  # If the Mountain Car environment should be rendered
IS_TRAINING = True  # If the agent is training
fig_format = 'png'  # Format used for saving matplotlib's figures

# Initiating the Mountain Car environment
env = gym.make('LunarLander-v2', render_mode='None')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Agent

# Configuração dos espaços de discretização
bins = [
    np.linspace(-1.5, 1.5, 15),         # xpos
    np.linspace(-1.5, 1.5, 15),         # ypos
    np.linspace(-5, 5, 15),             # xvel
    np.linspace(-5, 5, 15),             # yvel
    np.linspace(-np.pi, np.pi, 15),     # angle
    np.linspace(-5, 5, 15),             # angular vel
    np.linspace(0, 1, 2),               # leg1
    np.linspace(0, 1, 2)                # leg2 
]
state_tuple = tuple([len(b) + 1 for b in bins])
# Instancia o discretizador de estado
state_discretizer = StateDiscretizer(bins)
batch_size = 32  # batch size used for the experience replay
return_history = []
agent = QLearning(state_tuple, action_size, 0.5, 0.1, 0.99, epsilon_decay=0.98, epsilon_min=0.01)


for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    state = state_discretizer.discretize(state)
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    done = False
    if episodes > 300 and IS_TRAINING:
        IS_TRAINING = False
    for time in range(1, 500):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Select action
        if IS_TRAINING:
            action = agent.get_exploratory_action(state)
        else:
            action = agent.get_greedy_action(state)

        # Take action, observe reward and new state
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = state_discretizer.discretize(next_state)

        # Making reward engineering to allow faster training
        # reward = reward_engineering(state[0], action, reward, next_state[0], done)
        # Appending this experience to the experience replay buffer
        # agent.append_experience(state, action, reward, next_state, done)
        if IS_TRAINING:
                next_action = agent.get_exploratory_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
        state = next_state
        action = next_action
        # Accumulate reward
        cumulative_reward += reward

        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break

    return_history.append(cumulative_reward)
    agent.update_epsilon()

    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.pause(0.1)
        plt.savefig('agent.' + fig_format)

