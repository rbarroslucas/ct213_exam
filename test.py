import gym
import numpy as np
import warnings
from reinforcement_learning import Sarsa, QLearning

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2", render_mode="human")

state_size = env.observation_space.shape
num_actions = env.action_space.n

xpos_space = np.linspace(-1.5, 1.5, 15)
ypos_space = np.linspace(-1.5, 1.5, 15)
xvel_space = np.linspace(-5, 5, 15)
yvel_space = np.linspace(-5, 5, 15)
angle_space = np.linspace(-np.pi, np.pi, 15)
angular_vel_space = np.linspace(-5, 5, 15)
leg1_space = np.linspace(0, 1, 2)
leg2_space = np.linspace(0, 1, 2)

states_tuple = (len(xpos_space) + 1, len(ypos_space) + 1, len(xvel_space) + 1, len(yvel_space) + 1, len(angle_space) + 1, len(angular_vel_space) + 1, len(leg1_space), len(leg2_space))

def discretize_state(state):
    state_x = np.digitize(state[0], xpos_space)
    state_y = np.digitize(state[1], ypos_space)
    state_xvel = np.digitize(state[2], xvel_space)
    state_yvel = np.digitize(state[3], yvel_space)
    state_angle = np.digitize(state[4], angle_space)
    state_angular_vel = np.digitize(state[5], angular_vel_space)
    state_leg1 = np.digitize(state[6], leg1_space) - 1
    state_leg2 = np.digitize(state[7], leg2_space) - 1
    return (state_x, state_y, state_xvel, state_yvel, state_angle, state_angular_vel, state_leg1, state_leg2)

def run(num_episodes, algorithm, is_training=True, epsilon=0.1, alpha=0.1, gamma=0.99):
    if algorithm == 'sarsa':
        rl_algorithm = Sarsa(states_tuple, num_actions, epsilon, alpha, gamma)
    elif algorithm == 'qlearning':
        rl_algorithm = QLearning(states_tuple, num_actions, epsilon, alpha, gamma)

    for i in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state)
        reward_ep = 0
        done = False
        while not done:
            if is_training:
                action = rl_algorithm.get_exploratory_action(state)
            else:
                action = rl_algorithm.get_greedy_action(state)

            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)

            if is_training:
                next_action = rl_algorithm.get_exploratory_action(next_state)
                rl_algorithm.learn(state, action, reward, next_state, next_action)


            reward_ep += reward

            state = next_state
            action = next_action

        print('Episode: {0}, Reward: {1}'.format(i, reward_ep))


run(1000, 'qlearning', is_training=True)