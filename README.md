# Lunar Lander - CT213 Exam

This project uses the Lunar Lander environment from OpenAI's Gym library.

Group:

Lucas Ribeiro do Rêgo Barros (2°FUND - leader)

Luiz Felipe Vezzali de Paula (1° AER)

Mateus Pereira Alves (1° AESP)

## Project goal

This project aims to solve the Lunar Lander environment with DQN, Double DQN and Dueling DQN algorithms. The project's paper can be found between the files.

## Folder `Results`

The folder `Results` presents some trained agents with the raw results

#### 

## Environment Setup

To set up the Conda environment, follow these steps:

1.  Create the Conda environment using the existing one:
    ```sh
    conda env create -f exam_env.yml
    ```

3. Activate the environment:
    ```sh
    conda activate exam_env
    ```

## Running the Project

After activating the Conda environment, you can run the main code that :

```sh
python src/main.py
```

## Testing different agents

You can test different agents by choosing the configurations on `src/utils/params.py` 


## Evaluating a pre-trained agent

If you don't wanna train a agent and just evaluate, you can run the script `src/only_evaluation.py`

