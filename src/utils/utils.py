import os
import csv
import pickle
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


def create_folder(algorithm):
    """
    Creates a folder to save training results.

    :return: The name of the created folder.
    :rtype: str
    """
    existing_folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith(algorithm)]
    numbers = [int(f.split('#')[1]) for f in existing_folders]
    folder_number = max(numbers) + 1 if numbers else 1
    folder_name = f"{algorithm}#{folder_number}"
    os.makedirs(folder_name)
    return folder_name

def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("Nenhuma GPU disponível. Usando a CPU.")

def load_weights(agent):  
    if os.path.exists('lunar_lander_dqn.h5'):
        print('Loading weights from previous learning session.')
        agent.load_model("lunar_lander_dqn.h5")
    else:
        print('No weights found from previous learning session.')

def get_architecture(algorithm):
    if algorithm == 'DQN':
        return 'DQN 256/128'
    elif algorithm == 'DoubleDQN':
        return 'DoubleDQN 128/128'
    elif algorithm == 'DuelDQN':
        return 'DuelDQN 256/128'

# Determine the next folder number
def get_next_folder_number(base_name):
    existing_folders = [f for f in os.listdir() if os.path.isdir(f) and f.startswith(base_name)]
    numbers = [int(f.split('#')[1]) for f in existing_folders]
    return max(numbers) + 1 if numbers else 1

def mid_plots(algorithm, return_history, loss_history, agent, folder_name, fig_format):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(return_history)), y=return_history, label='Score', color='Gray')

    if len(return_history) >= 100:
        moving_average = pd.Series(return_history).rolling(window=100).mean()
        sns.lineplot(x=range(len(moving_average)), y=moving_average, color='r', label='100-episode Moving Average')

    plt.title(f'Training episodes score - {get_architecture(algorithm)}')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig(os.path.join(folder_name, 'return.' + fig_format))
    plt.close()

class FileHandler:
    def __init__(self, folder_name, algorithm):
        self.algorithm = algorithm

        self.return_history_path = os.path.join(folder_name, 'return_history.pkl')
        self.training_summary_path = os.path.join(folder_name, 'training_summary.csv')

    def save_return_history(self, return_history):
        with open(self.return_history_path, 'wb') as f:
            pickle.dump(return_history, f)

    def save_training_summary(self, num_ep, hyperparams, batch_size, elapsed_time, steps):
        gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, buffer_size = hyperparams
        training_info = {
            'Algorithm': self.algorithm,
            'Number of Episodes': num_ep,
            'Gamma': gamma,
            'Epsilon Initial': epsilon,
            'Epsilon Final': epsilon_min,
            'Epsilon Decay': epsilon_decay,
            'Learning Rate': learning_rate,
            'Batch Size': batch_size,
            'Buffer Size': buffer_size,
            'Elapsed Time (seconds)': elapsed_time,
            'Total Steps': steps,
        }

        with open(self.training_summary_path, 'w') as file:
            for key, value in training_info.items():
                file.write(f'{key}: {value}\n')