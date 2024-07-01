import numpy as np
import gym
import random
import matplotlib.pyplot as plt

from tensorflow.keras import models, layers, optimizers, activations, losses
from queue import Queue

NUM_EPISODES = 400  # Number of episodes used for training
RENDER = False  # If the Mountain Car environment should be rendered
IS_TRAINING = True  # If the agent is training
fig_format = 'png'  # Format used for saving matplotlib's figures

# Initiating the Mountain Car environment
env = gym.make(
    "LunarLander-v2",
    render_mode = "human",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power  = 1.5,
)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.95
epsilon = 0.5
epsilon_min = 0.01
epsilon_decay = 0.98
learning_rate = 0.001
buffer_size = 4098

# NN model
model = models.Sequential()
model.add(layers.Dense(256, input_dim=state_size, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(action_size, activation='linear'))
model.compile(loss=losses.mse, optimizer=optimizers.legacy.Adam(learning_rate=learning_rate))
model.summary()

# Sample Size and history
batch_size = 32  
return_history = []
replay_buffer = Queue(maxsize=buffer_size)

for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()

    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    done = False
    steps = 0
    if episode > 300 and IS_TRAINING:
        IS_TRAINING = False

    while not done:
        if RENDER:
            env.render() 
            
        if np.random.rand() < epsilon:
             np.random.randint(0, action_size)
        else:
            action = np.argmax(model.predict(state))

        # Take action, observe reward and new state
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        if replay_buffer.full():
            _ = replay_buffer.get()

        replay_buffer.put((state, action, reward, next_state, done))

        if replay_buffer.qsize() > 2 * batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states, targets = [], []
            for sample_state, sample_action, sample_reward, sample_next_state, sample_done in minibatch:
                target = model.predict(sample_state)
                if not done:
                    target[0][sample_action] = sample_reward + gamma * np.max(model.predict(sample_next_state)[0])
                else:
                    target[0][sample_action] = sample_reward
                # Filtering out states and targets for training
                states.append(sample_state[0])
                targets.append(target[0])
            history = model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
            # Keeping track of loss
            loss = history.history['loss'][0]

        cumulative_reward = gamma * cumulative_reward + reward


        state = next_state
        steps += 1


    print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}".format(episode, NUM_EPISODES, steps, cumulative_reward, epsilon))
    return_history.append(cumulative_reward)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    if episode % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.pause(0.1)
        plt.savefig('agent.' + fig_format)

