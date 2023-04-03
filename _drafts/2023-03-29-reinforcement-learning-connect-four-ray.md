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

<script	type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app src="https://clementbm-connectfour.hf.space"></gradio-app>


**Table of contents**
- [Technical Stack](#technical-stack)
  - [Poetry](#poetry)
  - [Ray \& RLlib](#ray--rllib)
  - [Petting Zoo](#petting-zoo)
- [Agent, Environment and Training](#agent-environment-and-training)
  - [Proximal Policy Optimization (PPO) Agent](#proximal-policy-optimization-ppo-agent)
  - [RLlib wrapper around PettingZoo Connect Four](#rllib-wrapper-around-pettingzoo-connect-four)
  - [RLlib Action Mask](#rllib-action-mask)
  - [Opponents](#opponents)
  - [RLlib Training Loop](#rllib-training-loop)
- [Integration and deployment (free of charge)](#integration-and-deployment-free-of-charge)
  - [Gradio App](#gradio-app)
  - [Export policy model as ONNX](#export-policy-model-as-onnx)
  - [Create a Space at Hugging Face](#create-a-space-at-hugging-face)
  - [Git LFS](#git-lfs)
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
`Algorithm` class is the cornerstone of RLlib components. The `Algorithm` has a policy, a neural network model that tries to maximize the expected sum over all future rewards.
The model/policy maps an observation of the environment to an action $$\Pi(s(t)) \rightarrow a(t)$$.

An `Algorithm` also sets up its `rollout workers` and `optimizers`, and collects training metrics. In RLlib a `rollout` is an episode, or a game of connect four in our case.

The following diagram illustrates the iterative learning process ([from RLlib documentation](https://docs.ray.io/en/latest/rllib/core-concepts.html)):

![alt](/assets/2023-03-29/rllib-key-concepts.png)


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
OpenAI made the new in April 2019 with their AI agent OpenAI Five, based on PPO, which defeated the world champions at the multiplayer game Dota 2.

## RLlib wrapper around PettingZoo Connect Four
Connect Four is a 2-player turn-based game. The players drop their tokens in a column of a standing grid, where each token fall until it reaches the bottom of the column or is stopped by an existing token. Players have to connect four of their tokens vertically, horizontally or diagonally. The game ends when a player has made a sequence of 4 tokens or when all columns have been filled, or if a player makes an illegal move.

The [PettingZoo Connect Four game environment](https://pettingzoo.farama.org/environments/classic/connect_four/) has to be wrapped before using it with RLlib API.

Once it's converted into an [rllib MultiAgentEnv](https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-multi-agent-environments) we can finally train our agents.

The connect four environment is described as follow:

| Key | Value | Description |
|--|--|--|
| Agent | ["player_0", "player_1"] | Two players. |
| Action | [0..6] | The list of all possible actions, the action space. |
| Observation | (6, 7, 2) [0,1] | A complete description of the environment, nothing is hidden in the state space. In this game, what's the players see (the observation state) is the same as the state space. Nothing is hidden. |
| Reward | [-1,0,1] | The reward is the feedback the agent receives per action. In this game, rewards are sparse, which means they are given when the game ends. Winner receives 1, looser receives -1, the ties get 0, cheater gets -1 and the other 0 |

Note that the wrapper has some important limitations:
1. All agents have the same `action_spaces` and `observation_spaces`.
2. Environments are positive sum games (-> Agents are expected to cooperate to maximize reward). This isn't a hard restriction, it just that standard algorithms aren't expected to work well in highly competitive games.

```python
from pettingzoo.classic import connect_four_v3

# define how to make the environment
env_creator = lambda config: connect_four_v3.env(render_mode="rgb_array")

# register the environment under an rllib name
register_env("connect4", lambda config: Connect4Env(env_creator(config)))
```

Overrides `reset()` method of [PettingZooEnv](https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/pettingzoo_env.py):

```python
@PublicAPI
class Connect4Env(PettingZooEnv):
  """Inherits from PettingZooEnv wrapper"""
  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    # In base class =>
    # info = self.env.reset(seed=seed, return_info=True, options=options)
    info = self.env.reset(seed=seed, options=options)
    return (
      {self.env.agent_selection: self.env.observe(self.env.agent_selection)},
      info or {},
    )
  def render(self):
    # In base class =>
    # return self.env.render(self.render_mode)
    return self.env.render()
```

## RLlib Action Mask
Players cannot place a token in a full column, and 

[Legal action mask](https://pettingzoo.farama.org/environments/classic/connect_four/#legal-actions-mask)
The legal moves available to the current agent are found in the action_mask element of the dictionary observation. The action_mask is a binary vector where each index of the vector represents whether the action is legal or not. The action_mask will be all zeros for any agent except the one whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

[TorchActionMaskModel](https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py#L69)

```python
class Connect4MaskModel(TorchModelV2, nn.Module):
  def __init__(
    self,
    obs_space,
    action_space,
    num_outputs,
    model_config,
    name,
    **kwargs,
  ):
    orig_space = getattr(obs_space, "original_space", obs_space)

    TorchModelV2.__init__(
      self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    )
    nn.Module.__init__(self)

    self.internal_model = TorchFC(
      orig_space["observation"],
      action_space,
      num_outputs,
      model_config,
      name + "_internal",
    )

  def forward(self, input_dict, state, seq_lens):
    # Extract the available actions tensor from the observation.
    action_mask = input_dict["obs"]["action_mask"]

    # Compute the unmasked logits.
    logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

    # Convert action_mask into a [0.0 || -inf]-type mask.
    inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
    masked_logits = logits + inf_mask

    # Return masked logits.
    return masked_logits, state

  def value_function(self):
    return self.internal_model.value_function()
```

## Opponents
```python
class AlwaysSameHeuristic(HeuristicBase):
  """
  Pick a random move and stick with it for the entire episode.
  """
  _rand_choice = random.choice(range(7))

  def _do_compute_actions(self, obs_batch):
    def select_action(legal_action):
      legal_choices = np.arange(len(legal_action))[legal_action == 1]

      if self._rand_choice not in legal_choices:
        self._rand_choice = np.random.choice(legal_choices)

      return self._rand_choice
    return [select_action(x) for x in obs_batch["action_mask"]], [], {}
```

Playing against old version of itself


Adding with the help of a CallbackFunction

```python

def create_self_play_callback(win_rate_thr, opponent_policies, opponent_count=10):
    class SelfPlayCallback(DefaultCallbacks):
        win_rate_threshold = win_rate_thr

        def __init__(self):
            # init logic ...

        def on_train_result(self, *, algorithm, result, **kwargs):
            """Called at the end of Algorithm.train().

            Args:
                algorithm: Current Algorithm instance.
                result: Dict of results returned from Algorithm.train() call.
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
            """
            main_rew = result["hist_stats"].pop("policy_learned_reward")
            opponent_rew = result["hist_stats"].pop("episode_reward")

            won = 0
            for r_main, r_opponent in zip(main_rew, opponent_rew):
                if r_main > r_opponent:
                    won += 1
            win_rate = won / len(main_rew)

            result["win_rate"] = win_rate
            print(f"Iter={algorithm.iteration} win-rate={win_rate}")

            if win_rate > self.win_rate_threshold:
                if len(self.opponent_policies) == self.opponent_policies.maxlen:
                    self.policy_to_remove = self.opponent_policies[0]

                new_pol_id = None
                while new_pol_id is None:
                    if np.random.choice(range(6)) == 0:
                        new_pol_id = np.random.choice(list(self.frozen_policies.keys()))
                    else:
                        self.current_opponent += 1
                        new_pol_id = f"learned_v{self.current_opponent}"

                    if new_pol_id in self.opponent_policies:
                        new_pol_id = None
                    else:
                        self.opponent_policies.append(new_pol_id)

                print("Non trainable policies", list(self.opponent_policies))

                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    return (
                        "learned"
                        if episode.episode_id % 2 == int(agent_id[-1:])
                        else np.random.choice(list(self.opponent_policies))
                    )

                print(
                    f"Iter={algorithm.iteration} Adding new opponent to the mix ({new_pol_id}). League size {len(self.opponent_policies) + 1}"
                )

                if new_pol_id in list(self.frozen_policies.keys()):
                    new_policy = algorithm.add_policy(
                        policy_id=new_pol_id,
                        policy_cls=self.frozen_policies[new_pol_id],
                        policy_mapping_fn=policy_mapping_fn,
                    )
                else:
                    new_policy = algorithm.add_policy(
                        policy_id=new_pol_id,
                        policy_cls=type(algorithm.get_policy("learned")),
                        policy_mapping_fn=policy_mapping_fn,
                    )
                    learned_state = algorithm.get_policy("learned").get_state()
                    new_policy.set_state(learned_state)
                algorithm.workers.sync_weights()

            else:
                print("Not good enough... Keep learning ...")

            result["league_size"] = len(self.opponent_policies) + 1

        def on_evaluate_end(self, *, algorithm, evaluation_metrics, **kwargs):
            """Runs when the evaluation is done.

            Runs at the end of Algorithm.evaluate().

            Args:
                algorithm: Reference to the algorithm instance.
                evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                    You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
            """

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                return (
                    "learned"
                    if episode.episode_id % 2 == int(agent_id[-1:])
                    else np.random.choice(list(self.opponent_policies))
                )

            if self.policy_to_remove is not None:
                print("Remove ", self.policy_to_remove, "from opponent policies")
                algorithm.remove_policy(
                    self.policy_to_remove,
                    policy_mapping_fn=policy_mapping_fn,
                )
                self.policy_to_remove = None
                algorithm.workers.sync_weights()

    return SelfPlayCallback

```

## RLlib Training Loop
Using `ray.tune`

```python
config = (
  (
    ppo.PPOConfig()
    .environment("connect4")
    .framework("torch")
    .training(model={"custom_model": Connect4MaskModel})
  )
  .multi_agent(
    policies={
      "learned": PolicySpec(),
      "always_same": PolicySpec(policy_class=AlwaysSameHeuristic),
      "linear": PolicySpec(policy_class=LinearHeuristic),
      "beat_last": PolicySpec(policy_class=BeatLastHeuristic),
      "random": PolicySpec(policy_class=RandomHeuristic),
    },
    policy_mapping_fn=select_policy,
    policies_to_train=["learned"],
  )
  .callbacks(
    create_self_play_callback(
      win_rate_thr=args.win_rate_threshold,
      opponent_policies=["always_same", "beat_last", "random", "linear"],
      opponent_count=15,
    )
  )
  .evaluation(evaluation_interval=1)
)
```

Ray Dashboard, monitoring the ray nodes, resources status (gpu, cpu, heap), graphana

[Prometheus setup](https://docs.ray.io/en/latest/ray-observability/ray-metrics.html)

```shell
results = tune.Tuner(
  "PPO",
  param_space=config.to_dict(),
  checkpoint_config=air.CheckpointConfig(
      num_to_keep=10,
      checkpoint_at_end=True,
      checkpoint_frequency=10,
      checkpoint_score_order="max",
  )
).fit()
results.get_best_result(metric="win_rate", mode="max")
```

Tensorboard
```shell
tensorboard --logdir ~/ray_results/
```

# Integration and deployment (free of charge)
## Gradio App
```shell
gradio connectfour/app.py
```
## Export policy model as ONNX

```python
config.evaluation(evaluation_interval=None)
```

```python
algo = Algorithm.from_checkpoint(analysis.best_checkpoint)
ppo_policy = algo.get_policy("learned")
ppo_policy.export_model("models", onnx=11)
```
`model.onnx`

Vizualize model with [NETRON](https://netron.app/)

## Create a Space at Hugging Face
```shell
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Git LFS
Do you have files larger than 10MB? Those files should be tracked with git-lfs, which you can initialize with:

e to git-lfs.github.com and click Download.
To install the file, run this command:
```shell
./install.sh
> Git LFS initialized.
```

Download and install the Git command line extension. Once downloaded and installed, set up Git LFS for your user account by running: 
```shell
git lfs install
```

When you use Hugging Face to create a repository, Hugging Face automatically provides a list of common file extensions for common Machine Learning large files in the .gitattributes file, which git-lfs uses to efficiently track changes to your large files. 

```shell
*.pkl filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
```

# Sources
