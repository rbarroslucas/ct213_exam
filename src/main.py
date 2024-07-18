import os
import tensorflow as tf

from utils.params import *
from utils.runner import Game_Runner
from utils.utils import set_gpu

'''Params already defined in src/utils/params.py'''

# Test the GPU, if wanna use it
if USE_GPU:
    set_gpu()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.compat.v1.disable_eager_execution()

# Create the Game Runner
runner = Game_Runner(ALGORITHM, MINIBATCH_SIZE, HYPERPARAMS)

# Train the agent
runner.train(TRAIN_EPISODES, NN_NAME, RENDER, fig_format)

# Evaluate the agent
runner.evaluate(TEST_EPISODES, load_agent=True, render=RENDER, fig_format=fig_format,nn_name=NN_NAME)