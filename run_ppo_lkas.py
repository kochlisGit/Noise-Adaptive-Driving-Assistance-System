import random
import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from matplotlib import pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.catalog import ModelCatalog
from nadas.environments.lkas_env import LKASEnvironment
from nadas.models.lstm_actor_critic import LSTMActorCritic
import time
import os


def main(
        server_ip: str,
        port: int,
        num_iterations: int,
        sensor_data_corrupt_prob: float,
        segmentation_noise_ratio: float,
        segmentation_corrupt_portion_size: tuple,
        depth_error_rate: float,
        use_state_prediction: bool,
        max_steps: int,
        iterations_per_reload: int,
        action_repeats: int,
        debug: bool,
        store_sensor_directory: str or None
):
    # Set the path to the Carla server executable
    CARLA_SERVER_PATH = '../CarlaUE4.exe'

    # Set the arguments for launching the Carla server
    CARLA_SERVER_ARGS = [
        '-quality-level=Low',
        '-carla-server',
        '-RenderOffscreen',
        '-ResX=800',
        '-ResY=600'
        # '-carla-host=155.207.113.68'
    ]

    print("Launching Carla server...")

    # Launch the Carla server process
    # carla_server_process = subprocess.Popen([CARLA_SERVER_PATH] + CARLA_SERVER_ARGS)

    checkpoint_dir = 'nadas/experiments/checkpoints/ppo'

    env_config = {
        'server_ip': server_ip,
        'port': port,
        'sensor_data_corrupt_prob': sensor_data_corrupt_prob,
        'segmentation_noise_ratio': segmentation_noise_ratio,
        'segmentation_corrupt_portion_size': segmentation_corrupt_portion_size,
        'depth_error_rate': depth_error_rate,
        'use_state_prediction': use_state_prediction,
        'max_steps': max_steps,
        'iterations_per_reload': iterations_per_reload,
        'action_repeats': action_repeats,
        'debug': debug,
        'store_sensor_directory': store_sensor_directory
    }

    ModelCatalog.register_custom_model('ppo_lstm_model', LSTMActorCritic)

    agent_config = PPOConfig()
    agent_config.model.update({
        'vf_share_layers': True,
        'custom_model': 'ppo_lstm_model',
        'custom_model_config': {},
        'use_lstm': True,
        'lstm_cell_size': 64,
        'max_seq_len': 10
    })
    agent_config.rollouts(num_rollout_workers=1, rollout_fragment_length=1024)
    # agent_config.framework(framework='tf2', eager_tracing=True)
    agent_config.batch_mode = 'complete_episodes'
    agent_config.use_critic = True
    agent_config.use_gae = True
    agent_config.clip_param = 0.2
    agent_config.entropy_coeff = 0.01
    agent_config.kl_coeff = 0.01
    agent_config.vf_loss_coeff = 0.5
    agent_config.shuffle_sequences = True
    agent_config.num_sgd_iter = 20
    agent_config.sgd_minibatch_size = 32
    agent_config.train_batch_size = 1024
    agent_config.seed = 0
    agent_config.gamma = 0.99
    agent_config.lr = 0.0005
    agent_config.num_gpus = 0
    agent_config.normalize_actions = True
    agent_config.disable_env_checking = True

    agent = agent_config.environment(env=LKASEnvironment, env_config=env_config).build()
    agent.save(checkpoint_dir=checkpoint_dir)

    # if summary:
    #     agent.get_policy().model.base_model.summary(expand_nested=True)
    #     agent.get_policy().model.rnn_model.summary(expand_nested=True)

    average_returns = []
    episode_steps = []
    checkpoints = []
    # logs = pd.read_csv('nadas/experiments/logs/ppo_train_integrated.csv')
    # average_returns = logs['Average Returns'].to_list()
    # episode_steps = logs['Episode Steps'].to_list()
    # checkpoints = logs['Checkpoint'].to_list()
    i = 0
    current_checkpoint = 0
    last_checkpoint = 0
    # agent = Algorithm.from_checkpoint('nadas/experiments/checkpoints/ppo/checkpoint_000000')

    while i < num_iterations:
        try:
            print(f'Training Iteration {i + 1}')

            result = agent.train()

            i += 1
            current_checkpoint += 1
            agent.save(checkpoint_dir=checkpoint_dir)

            returns = result['episode_reward_mean']
            average_returns.append(returns)

            steps = result['num_env_steps_trained_this_iter']
            episode_steps.append(steps)

            checkpoints.append(current_checkpoint)

            print(
                "# -----\n"
                f"Iter = {i}\n"
                f"Iter time         = {result['time_this_iter_s']}\n"
                f"Average Returns   = {returns}\n"
                f"Episode Steps     = {steps}\n\n"
            )

            df = pd.DataFrame({
                'Average Returns': average_returns,
                'Episode Steps': episode_steps,
                'Checkpoint': checkpoints
            })
            df.to_csv('nadas/experiments/logs/ppo_train_integrated.csv', index=False)

            fig, axes = plt.subplots(1, 1, figsize=(20, 10))
            fig.tight_layout()
            axes.plot(average_returns)
            axes.set_title('PPO - Average Returns')
            axes.set_ylabel('Returns')
            axes.set_xlabel('Iteration')
            fig.savefig(f'nadas/experiments/plots/ppo_average_returns2.png')
            plt.close()

            fig, axes = plt.subplots(1, 1, figsize=(20, 10))
            fig.tight_layout()
            axes.plot(episode_steps)
            axes.set_title('PPO - Episode Steps')
            axes.set_ylabel('Steps')
            axes.set_xlabel('Iteration')
            fig.savefig(f'nadas/experiments/plots/ppo_episode_steps.png')
            plt.close()

        except Exception as e:
            print(f'Exception raised: {e}')
            while True:
                try:
                    print("Carla server has crashed! Or TimeoutError Occurred!")
                    # winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)

                    os.system("taskkill /f /im  CarlaUE4.exe")
                    os.system("taskkill /f /im  CarlaUE4-Win64-Shipping.exe")
                    time.sleep(10)

                    ray.shutdown()
                    ray.init()
                    ModelCatalog.register_custom_model('ppo_lstm_model', LSTMActorCritic)

                    if current_checkpoint != 0:
                        ckp_id = str(current_checkpoint).zfill(6)
                    else:
                        ckp_id = str(last_checkpoint).zfill(6)
                    agent = Algorithm.from_checkpoint(f'nadas/experiments/checkpoints/ppo/checkpoint_{ckp_id}')
                    last_checkpoint = current_checkpoint
                    current_checkpoint = 0
                    break
                except Exception as ex:
                    print(ex)
                    continue


if __name__ == '__main__':
    server_ip = 'localhost'
    port = 2000
    train_iterations = 1000
    summary = True
    # scenario_dict = {'Town02_Opt': Town02(), 'Town03_Opt': Town03(), 'Town04_Opt': Town04(), 'Town10HD_Opt': Town10()}

    ray.shutdown()
    ray.init()

    tf.random.set_seed(seed=0)
    np.random.seed(0)
    random.seed(0)

    main(
        server_ip='localhost',
        port=2000,
        num_iterations=500,
        sensor_data_corrupt_prob=0.2,
        segmentation_noise_ratio=0.2,
        segmentation_corrupt_portion_size=(10, 15),
        depth_error_rate=0.1,
        max_steps=1024,
        use_state_prediction=True,
        iterations_per_reload=10,
        action_repeats=2,
        debug=False,
        store_sensor_directory=None
    )

    ray.shutdown()
