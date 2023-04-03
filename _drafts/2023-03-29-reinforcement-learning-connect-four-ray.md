---
layout: post
title:  "Reinforcement Learning with Ray and Connect Four"
excerpt: "Reiforcement learning ray, connect four, hugging face space, multi agents, torch, gradio"
date:   2023-03-29
categories: [project]
tags: [reinforcement learning, ray, hugging face, gradio, multi agents]
---

![Red Hot Peppers](/assets/2023-03-29/chili-g100503ee3_1280.jpg){: width="100%"  }

In this post I'll demonstrate how to train an agent to play [connect four](https://pettingzoo.farama.org/environments/classic/connect_four/) using [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning). Then I'll show you how to integrate it in a [gradio application](https://www.gradio.app/) and finally making it public by deploying it on [Hugging Face space](https://huggingface.co/docs/hub/spaces).

If you want to learn more about the underlying mechanisms of reinforcement learning, you'd better look somewhere else. A good starting point would be [Deep Reinforcement Learning by Udacity](https://github.com/udacity/deep-reinforcement-learning) course or the RL section in [Hands-On Machine Learning by Aurélien Geron](https://github.com/ageron/handson-ml3/blob/main/18_reinforcement_learning.ipynb). This being said, in the RL field, a (software) agent observes a virtual environment, and takes actions given the environment state. In return the environment assigns a reward to the agent. The objective of the agent is to learn to act in a way that will maximize its expected rewards over time.

With RL, an agent can be trained to:
* control a robot
* regulate house temperature as a thermostat
* drive a car in an intelligent transportation system (ITS)
* make recommandation in a recommender system

Reinforcement learning is different than traditional machine learning technics such as supervised or unsupervised learning. Supervised Learning tries to predict or classify y, given x. Unsupervised Learning tries to simplify y, given x. Whereas Reinforcement Learning tries to choose the "best" actions given an uncertain environment.

All the code for this project is available at [Hugging Face](https://huggingface.co/spaces/ClementBM/connectfour/tree/main), and you may be able to clone the repository with the following command:
```shell
git clone https://huggingface.co/spaces/ClementBM/connectfour
```


**Table of contents**
- [Technical Stack](#technical-stack)
  - [Poetry](#poetry)
  - [Ray \& RLlib](#ray--rllib)
  - [Petting Zoo](#petting-zoo)
- [Agent, Environment and Training](#agent-environment-and-training)
  - [Proximal Policy Optimization (PPO) Agent](#proximal-policy-optimization-ppo-agent)
  - [RLlib PettingZoo Wrapper](#rllib-pettingzoo-wrapper)
  - [RLlib Action Mask](#rllib-action-mask)
  - [RLlib Training Loop](#rllib-training-loop)
- [Integration and deployment (free of charge)](#integration-and-deployment-free-of-charge)
  - [Gradio App](#gradio-app)
  - [Export policy model as ONNX](#export-policy-model-as-onnx)
  - [Create a Space at Hugging Face](#create-a-space-at-hugging-face)
  - [Git LFS](#git-lfs)
- [Play against the agent](#play-against-the-agent)
- [Sources](#sources)


# Technical Stack

## [Poetry](https://python-poetry.org/)
`Poetry` is a packaging and dependency management system in python powered by [Vercel](https://vercel.com/). It makes virtual environments easy to handle and it's multi-platform: Linux, macOS and Windows.
You just declare the libraries your project depends on and `Poetry` manage installs and updates for you.

`Poetry` has also a lockfile to ensure repeatable installs (on your CI server, or other developer machines), and consistency when deployed on production machines.

## [Ray](https://www.ray.io/) & [RLlib](https://www.ray.io/rllib)
`Ray` is an open source framework powered by Anyscale that provides a simple, universal API for building distributed systems and tools to parallelize machine learning workflows. The framework is a large ecosystem of applications, libraries and tools dedicated to machine learning. It also integrates with [multiple libraries for distributed execution](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html): scikit-learn, XGBoost, TensorFlow, PyTorch, Hugging Face, spaCy, LightGBM, Horovod, ...

`RLlib` is part of the `Ray` ecosystem as a reinforcement learning library. It offers high scalability and a unified API for a variety of applications. `RLlib` natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic. `RLlib` has a huge number of state-of-the-art RL algorithms implemented.

Some well known industry actors already use `Ray`:
* at Microsoft, [ChatGPT developer OpenAI is using Ray](https://www.datanami.com/2023/02/10/anyscale-bolsters-ray-the-super-scalable-framework-used-to-train-chatgpt/) for training large language models
* at [Shopify](https://shopify.engineering/merlin-shopify-machine-learning-platform) to build Merlin, the Shopify's machine learning platform
* at [Uber](https://drive.google.com/file/d/1BS5lfXfuG5bnI8UM6FdUrR7CiSuWqdLn/view) for large scale deep learning training and tuning

## [Petting Zoo](https://pettingzoo.farama.org/)
`PettingZoo` is a Python library for conducting research in multi-agent reinforcement learning. It's powered by the [Farama foundation](https://github.com/Farama-Foundation), a nonprofit organization working to develop and maintain open source reinforcement learning tools.

It has different types of environment:
* Multi-player Atari 2600 games (cooperative, competitive and mixed sum)
* Classic: Classical games including card games, board games, etc.
* ...

# Agent, Environment and Training
`Algorithm` class is the cornerstone RLlib components. Each `Algorithm` subclass is managed by a `AlgorithmConfig`. An Algorithm sets up its `rollout workers` and `optimizers`, and collects training metrics.

The model that tries to maximize the expected sum over all future rewards is called a policy.
The policy is a function mapping the environment’s observations to an action to take, usually written $$\Pi(s(t)) -> a(t)$$. Following diagram illustrate the iterative learning process.

![alt](/assets/2023-03-29/rllib-key-concepts.png)

The simulation iterations of action -> reward -> next state -> train -> repeat, until the end state, is called an episode, or in RLlib, a `rollout`.

## [Proximal Policy Optimization (PPO) Agent](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

[paper](https://arxiv.org/abs/1707.06347) PPO’s clipped objective supports multiple SGD passes over the same batch of experiences. RLlib’s multi-GPU optimizer pins that data in GPU memory to avoid unnecessary transfers from host memory, substantially improving performance over a naive implementation. PPO scales out using multiple workers for experience collection, and also to multiple GPUs for SGD.

The mental model for multi-agent in RLlib is as follows:
1. Your environment (a sub-class of MultiAgentEnv) returns dictionaries mapping agent IDs (e.g. strings; the env can chose these arbitrarily) to individual agents’ observations, rewards, and done-flags.
2. You define (some of) the policies that are available up front (you can also add new policies on-the-fly throughout training), and
3. You define a function that maps an env-produced agent ID to any available policy ID, which is then to be used for computing actions for this particular agent.

https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical

Policy
![model](/assets/2023-03-29/model.onnx.png){: width="300" style="float: right; padding:15px"  }

Actor-Critic algorithm: a family of RL algorithms that combine Policy Gradients with Deep Q-Networks. An Actor-Critic agent contains two neural networks: a policy net and a DQN. The DQN is trained normally, by learning from the agent's experiences. The policy net learns differently (and much faster), than in regular Policy Gradient: instead of estimating the value of each action by going through multiple episodes, then summing the future discounted rewards for each action, and finalyy normalizing them, the agent (actor) relies on the action values estimated by the DQN (critic). It's a bit like an athlete (the agent) learning with the help of a coach (the DQN).

A2C: multiple agents learn in parallel, exploring different copies of the environment. At regular intervals, each agent pushes some weight updates to a master network, then it pulls the latest weights from that network. Each agent thus contributes to improving the master network and benefits from what the other agents have learned. Moreover, instead of estimating the Q-Values, the DQN estimates the advantage ot each action which stabilies training. All model updates are synchronous, so gradient updates are perormed over larger batches, which allows the model to better utilize the power of the GPU.

PPO: based on A2C that clips the loss function to avoid excessively large weight updates (which often lea to training instabilities). PPO is a simplification of the previous TRPO algorithm.
OpenAI made the new in April 2019 with their AI agent OpenAI Five, based on PPO, which defeated the world champions at the multiplayer fame Dota 2.

## RLlib PettingZoo Wrapper
An RLlib environment consists of:

* all possible actions (action space)
* a complete description of the environment, nothing hidden (state space)
* an observation by the agent of certain parts of the state (observation space)
* reward, which is the only feedback the agent receives per action.


Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing token. Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.

| Key | Value |
|--|--|
| Agents | ['player_0', 'player_1'] |
| Action Shape | (1,) |
| Action Values | Discrete(7) |
| Observation Shape | (6, 7, 2) |
| Observation Values | [0,1] |

turn-based games over environments, 

Sparse rewards are infrequent as the name implies. Sparse rewards could be given only after many steps, say when an agent wins a game, or completes a desired task.
Deterministic rewards are those rewards that always occur when an agent reaches a certain state and a certain control is taken.

https://pettingzoo.farama.org/environments/classic/connect_four/

https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-multi-agent-environments

## RLlib Action Mask
[Legal action mask](https://pettingzoo.farama.org/environments/classic/connect_four/#legal-actions-mask)
The legal moves available to the current agent are found in the action_mask element of the dictionary observation. The action_mask is a binary vector where each index of the vector represents whether the action is legal or not. The action_mask will be all zeros for any agent except the one whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

## RLlib Training Loop
Using ray tune

Hyperparameter tuning

Tensorboard
```shell
tensorboard --logdir {logdir}
```

Ray Dashboard, monitoring the ray nodes, resources status (gpu, cpu, heap), graphana


# Integration and deployment (free of charge)
## Gradio App

## Export policy model as ONNX

## Create a Space at Hugging Face

## Git LFS


# Play against the agent

<script	type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app src="https://clementbm-connectfour.hf.space"></gradio-app>

# Sources
