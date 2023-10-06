# Noise-Adaptive-Driving-Assistance-System
Noise-Adaptive Driving Assistance System (NADAS) using Deep Reinforcement Learning, State-Estimation &amp; State Representation. In this project, we construct an Adaptive-Cruise-Controller for the CARLA environments, 
using Deep Reinforcement Learning. We also include the files for the implementation of a Lane-Keep-Assistance (LKAS), which however is still in experimental phase.

Artificial Noise and data corruption has been added, in order to make a challenging driving environment, as well as increase the generalization cabability of the agents. To address noisy measurements and data corruption, we 
emply three techniques:

1. A State-Estimation algorithm
2. A State-Representation method using Stacked-States
3. A State-Representation method using LSTM as an alternative

The learning algorithm of the ACC controller is Proximal-Policy-Optimization (PPO) (https://arxiv.org/abs/1707.06347), with the option of adding LSTM neural networks.
To ensure the feasability of our approach, we also conducted response time benchmarks on Raspberry Pi 4. 

# Perception

The vehicle uses 2 front cameras: A semantic segmentation camera and a depth camera. These are manually calibrated in order to have the same focus, field of view and perspective. Then, the frames of the 2 images as combined.

![Perception](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/frames.jpeg)

# Paper
(To be submitted)

# Proximal-Policy-Optimization 

![PPO](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/ppo.jpeg)

# State-Estimation

![State-Estimation](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/stateestim.jpeg)

# State-Representation (State-Stacking / LSTMs)

![State-Stacking](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/staterepr.jpeg)

# NADAS

![NADAS](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/nadas.jpeg)

# ACC Experiment Results

![NADAS](https://github.com/kochlisGit/Noise-Adaptive-Driving-Assistance-System/blob/main/experiments.jpeg)

# Requirements

You need to download Carla 0.9.13, as well as Python 3.7. The python version is important, as this version of CARLA works only with Python 3.7. Additionally, you will need to 
download and install the following requirements:

* Python 3.7
* Carla 0.9.13
* Tensorflow 2.9.1
* RLLib 2.3.1
* Matplotlib
* Numpy
* Notebook
* Gym or Gymnasium
* Pygame
* Scikit-Learn
* Tensorflow Directml Plugin (optional)

You can easily install these requirements via `pip install` or `conda install`, depending on your package manager. 

# Training

To train the agents, you can run one of the following files (`e.g. python file.py`)

* run_ppo_acc.py : Train PPO with/without LSTMs
* run_ppo_acc_stacking.py : Train PPO with/without state-stacking
* run_ppo_lkas.py : Train PPO with/without LSTMs
* run_ppo_lkas_stacking.py : Train PPO with/without LSTMs
