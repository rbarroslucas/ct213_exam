# Select the algorithm to be used
# 'DQN' or 'DoubleDQN' or 'DuelDQN'
ALGORITHM = 'DQN'

# Set visual and gpu parameters
RENDER_MODE = None # 'human' or None
USE_GPU = False
RENDER = False
TRAIN_EPISODES = 2000
TEST_EPISODES = 100
fig_format = 'png'

ID = 'LunarLander-v2'


# Hyperparameters
MINIBATCH_SIZE = 64
buffer_size = 65536
gamma = 0.99
learning_rate = 0.0001
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998

# Do not change the following lines
HYPERPARAMS = (gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, buffer_size)
NN_NAME = f'{ALGORITHM}_LunarLander.h5'