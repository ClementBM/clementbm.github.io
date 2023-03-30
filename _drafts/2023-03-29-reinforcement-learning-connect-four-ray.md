---
layout: post
title:  "Reinforcement Learning with Ray and Connect Four"
excerpt: "Reiforcement learning ray, connect four, hugging face space, multi agents, torch, gradio"
date:   2023-03-29
categories: [project]
tags: [reinforcement learning, ray, hugging face, gradio, multi agents]
---

![Red Hot Peppers](/assets/2023-03-29/chili-g100503ee3_1280.jpg){: width="100%"  }

In this post I'll demonstrate how to train an agent to play connect four using reinforcement learning. Then I'll show you how to integrate it in a gradio application and finally making it public by deploying it on hugging face space.

If you want to learn more about the underlying mechanisms of reinforcement learning, you'd better look at another post. This being said, just quick remainder of the basic concept. In reinforcement learning (RL), an agent makes observations and takes actions within an environment, and in return it receives rewards. Its objective is to learn to act in a way that will maximize its expected rewards over time.
With RL, an agent can be trained to
* control a robot
* regulate house temperature, thermostat
* drive a car, intelligent transportation system (ITS)
* make recommandation, recommender system

Reinforcement learning is different than traditional machine learning such as supervised or unsupervised learning:
* Supervised learning try to predict y, given x 
* Unsupervised learning try to simplify y, given x

- [Technical Stack](#technical-stack)
  - [Poetry](#poetry)
  - [Ray by Anyscale](#ray-by-anyscale)
  - [Petting Zoo](#petting-zoo)
- [Agent, Environment and Training](#agent-environment-and-training)
  - [RLlib PPO](#rllib-ppo)
  - [RLlib Multi Agent](#rllib-multi-agent)
  - [RLlib PettingZoo Wrapper](#rllib-pettingzoo-wrapper)
  - [RLlib Action Mask](#rllib-action-mask)
  - [RLlib Training Loop](#rllib-training-loop)
- [Deploy (free of charge)](#deploy-free-of-charge)
  - [Gradio App](#gradio-app)
  - [Create a Space at Hugging Face](#create-a-space-at-hugging-face)
  - [Git LFS](#git-lfs)
- [Play against the agent](#play-against-the-agent)
- [Sources](#sources)


# Technical Stack

## [Poetry](https://python-poetry.org/)
Python packaging and dependency management made easy 
Having an insight of your project's dependencies is just one command away.
Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution.

Poetry requires Python 3.7+. It is multi-platform and the goal is to make it work equally well on Linux, macOS and Windows.

Committing this file to VC is important because it will cause anyone who sets up the project to use the exact same versions of the dependencies that you are using. Your CI server, production machines, other developers in your team, everything and everyone runs on the same dependencies, which mitigates the potential for bugs affecting only some parts of the deployments. Even if you develop alone, in six months when reinstalling the project you can feel confident the dependencies installed are still working even if your dependencies released many new versions since then. (See note below about using the update command.)

Powered by [Vercel](https://vercel.com/).

## Ray by Anyscale
RLlib is an open-source library for reinforcement learning (RL), offering support for production-level, highly distributed RL workloads while maintaining unified and simple APIs for a large variety of industry applications. Whether you would like to train your agents in a multi-agent setup, purely from offline (historic) datasets, or using externally connected simulators, RLlib offers a simple solution for each of your decision making needs.

RLlib does not automatically install a deep-learning framework, but supports TensorFlow as well as PyTorch.

ChatGPT developer OpenAI is using Ray, an open-source unified compute framework, to ease the infrastructure costs and complexity of training its large language models. Anyscale, the company behind Ray, 

https://www.datanami.com/2023/02/10/anyscale-bolsters-ray-the-super-scalable-framework-used-to-train-chatgpt/

Also use by Shopify...
https://shopify.engineering/merlin-shopify-machine-learning-platform

By Uber
https://drive.google.com/file/d/1BS5lfXfuG5bnI8UM6FdUrR7CiSuWqdLn/view

Ray is an open source framework that provides a simple, universal API for building distributed systems and tools to parallelize machine learning workflows. Ray is a large ecosystem of applications, libraries and tools dedicated to machine learning such as distributed scikit-learn, XGBoost, TensorFlow, PyTorch, Hugging Face, spaCy, LightGBM, Horovod

When using Ray, you get a cluster that enables you to distribute your computation across multiple CPUs and machines. In the following example, we train a model using Ray:

![alt](/assets/2023-03-29/rllib-key-concepts.png)

The simulation iterations of action -> reward -> next state -> train -> repeat, until the end state, is called an episode, or in RLlib, a rollout.


## [Petting Zoo](https://pettingzoo.farama.org/environments/classic/connect_four/)

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

# Agent, Environment and Training
## [RLlib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)
Proximal Policy Optimization (PPO)
[paper](https://arxiv.org/abs/1707.06347) PPO’s clipped objective supports multiple SGD passes over the same batch of experiences. RLlib’s multi-GPU optimizer pins that data in GPU memory to avoid unnecessary transfers from host memory, substantially improving performance over a naive implementation. PPO scales out using multiple workers for experience collection, and also to multiple GPUs for SGD.

## RLlib Multi Agent
The mental model for multi-agent in RLlib is as follows: (1) Your environment (a sub-class of MultiAgentEnv) returns dictionaries mapping agent IDs (e.g. strings; the env can chose these arbitrarily) to individual agents’ observations, rewards, and done-flags. (2) You define (some of) the policies that are available up front (you can also add new policies on-the-fly throughout training), and (3) You define a function that maps an env-produced agent ID to any available policy ID, which is then to be used for computing actions for this particular agent.

https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical

## RLlib PettingZoo Wrapper
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


# Deploy (free of charge)
## Gradio App

## Create a Space at Hugging Face

## Git LFS

# Play against the agent

<script	type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app src="https://clementbm-connectfour.hf.space"></gradio-app>

# Sources
