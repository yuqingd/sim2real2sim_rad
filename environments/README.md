# Meta-World
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/rlworkgroup/metaworld.svg?branch=master)](https://travis-ci.com/rlworkgroup/metaworld)

__Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.__ We aim to provide task distributions that are sufficiently broad to evaluate meta-RL algorithms' generalization ability to new behaviors.

For more background information, please refer to our [website](https://meta-world.github.io) and the accompanying [conference publication](https://arxiv.org/abs/1910.10897), which **provides baseline results for 8 state-of-the-art meta- and multi-task RL algorithms**.

If you would like very infrequent announcements about the status of the benchmark, critical bugs and known issues before conference deadlines, and future plans, please join our mailing list: [metaworld-announce@googlegroups.com](https://groups.google.com/forum/#!forum/metaworld-announce).

__Table of Contents__
- [Installation](#installation)
- [Using the benchmark](#using-the-benchmark)
  * [Basics](#basics)
  * [Running ML1](#running-ml1)
  * [Running ML10 and ML45](#running-ml10-and-ml45)
  * [Running MT10 and MT50](#running-mt10-and-mt50)
  * [Running Single-Task Environments](#running-single-task-environments)
- [Citing Meta-World](#citing-meta-world)
- [Become a Contributor](#become-a-contributor)
- [Acknowledgements](#acknowledgements)

## Installation
Meta-World is based on MuJoCo, which has a proprietary dependency we can't set up for you. Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) in the mujoco-py package for help. Once you're ready to install everything, run:

```
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
```

Alternatively, you can clone the repository and install an editable version locally:

```
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld
pip install -e .
```

## Using the benchmark
Here is a list of benchmark environments for meta-RL (ML*) and multi-task-RL (MT*):
* [__ML1__](https://meta-world.github.io/figures/ml1.gif) is a meta-RL benchmark environment which tests few-shot adaptation to goal variation within single task. You can choose to test variation within any of [50 tasks](https://meta-world.github.io/figures/ml45-1080p.gif) for this benchmark.
* [__ML10__](https://meta-world.github.io/figures/ml10.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 10 meta-train tasks, and 3 test tasks.
* [__ML45__](https://meta-world.github.io/figures/ml45-1080p.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 45 meta-train tasks and 5 test tasks.
* [__MT10__](https://meta-world.github.io/figures/mt10.gif) and __MT50__ are multi-task-RL benchmark environments for learning a multi-task policy that perform 10 and 50 training tasks respectively. MT10 and MT50 augment environment observations with a one-hot vector which identifies the task.


### Basics
We provide a `Benchmark` API, that allows constructing environments following the [`gym.Env`](https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/core.py#L8) interface.

To use a `Benchmark`, first construct it (this samples the tasks allowed for one run of an algorithm on the benchmark).
Then, construct at least one instance of each environment listed in `benchmark.train_classes` and `benchmark.test_classes`.
For each of those environments, a task must be assigned to it using
`env.set_task(task)` from `benchmark.train_tasks` and `benchmark.test_tasks`,
respectively.
`Tasks` can only be assigned to environments which have a key in
`benchmark.train_classes` or `benchmark.test_classes` matching `task.env_name`.

Please see below for some small examples using this API.


### Running ML1
```python
import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v1') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v1']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
```

### Running a benchmark:
Create an environment with train tasks (ML10, MT10, ML45, or MT50):
```python
import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
```
Create an environment with test tasks (this only works for ML10 and ML45, since MT10 and MT50 don't have a separate set of test tasks):
```python
import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

testing_envs = []
for name, env_cls in ml10.test_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.test_tasks
                        if task.env_name == name])
  env.set_task(task)
  testing_envs.append(env)

for env in testing_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
```

## Citing Meta-World
You use Meta-World for academic research, please kindly cite our CoRL 2019 paper the using following BibTeX entry.

```
@inproceedings{yu2019meta,
  title={Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author={Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2019}
  eprint={1910.10897},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
  url={https://arxiv.org/abs/1910.10897}
}
```

## Become a Contributor
We welcome all contributions to Meta-World. Please refer to the [contributor's guide](https://github.com/rlworkgroup/metaworld/blob/master/CONTRIBUTING.md) for how to prepare your contributions.

## Acknowledgements
Meta-World is a work by [Tianhe Yu (Stanford University)](https://cs.stanford.edu/~tianheyu/), [Deirdre Quillen (UC Berkeley)](https://scholar.google.com/citations?user=eDQsOFMAAAAJ&hl=en), [Zhanpeng He (Columbia University)](https://zhanpenghe.github.io), [Ryan Julian (University of Southern California)](https://ryanjulian.me), [Karol Hausman (Google AI)](https://karolhausman.github.io),  [Chelsea Finn (Stanford University)](https://ai.stanford.edu/~cbfinn/) and [Sergey Levine (UC Berkeley)](https://people.eecs.berkeley.edu/~svlevine/).

The code for Meta-World was originally based on [multiworld](https://github.com/vitchyr/multiworld), which is developed by [Vitchyr H. Pong](https://people.eecs.berkeley.edu/~vitchyr/), [Murtaza Dalal](https://github.com/mdalal2020), [Ashvin Nair](http://ashvin.me/), [Shikhar Bahl](https://shikharbahl.github.io), [Steven Lin](https://github.com/stevenlin1111), [Soroush Nasiriany](http://snasiriany.me/), [Kristian Hartikainen](https://hartikainen.github.io/) and [Coline Devin](https://github.com/cdevin). The Meta-World authors are grateful for their efforts on providing such a great framework as a foundation of our work. We also would like to thank Russell Mendonca for his work on reward functions for some of the environments.
