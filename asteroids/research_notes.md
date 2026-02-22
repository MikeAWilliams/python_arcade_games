# Research Notes: AI for Asteroids

> Compiled from web research, February 2026. Links verified at time of writing.

---

## General Summary

A lot of people have tried this. The field splits cleanly into two populations:

**Pixel-based deep RL** (DQN, PPO, A3C, IMPALA) — treats the game as a vision problem, uses CNNs, and
generally produces mediocre results on Asteroids specifically. DQN is one of the *worst* performing algorithms
on this game. The flickering problem (the Atari 2600 hardware only renders asteroids on alternating frames) kills
naive DQN implementations outright. Once you fix the implementation bugs, larger RL algorithms do eventually
work — IMPALA reaches superhuman — but they need enormous amounts of compute.

**Hand-crafted state + neuroevolution** (NEAT, genetic algorithms) — uses compact geometric features like
raycasting distances, and evolves small network topologies. These tend to work well for Asteroids with very
little compute because the state representation does most of the heavy lifting. The "Six Neurons" paper
showed 6–18 neurons can match large CNNs given a good enough encoding.

**Behavioral cloning on Asteroids specifically** is almost entirely unstudied. The closest academic work is
about imbalanced datasets in general imitation learning, which confirms our problem but offers no
Asteroids-specific solution.

**The single most useful insight:** every project that successfully trained a small network pre-computed
geometric features (distance to nearest asteroid, angular alignment, raycasting) before the network saw
anything. No successful small-network project feeds raw `(x, y, vx, vy)` coordinates per object and expects
the network to learn geometry from scratch. We do, and that is likely a core reason our model struggles.

---

## Benchmark Scores (for reference)

From [endtoend.ai](http://www.endtoend.ai/envs/gym/atari/asteroids/) — Atari Asteroids across all published algorithms:

### No-op Starts
| Algorithm | Score |
|-----------|-------|
| IMPALA (deep) | 108,590 |
| NoisyNet DuDQN | 86,700 |
| Human | 47,389 |
| ACKTR | 34,172 |
| NoisyNet A3C | 4,541 |
| QR-DQN-1 | 4,226 |
| IMPALA (shallow) | 3,508 |
| NoisyNet DQN | 3,455 |
| Rainbow | 2,713 |
| PPO | 2,098 |
| DQN | 1,629 |
| Random | 719 |

### Human Starts
| Algorithm | Score |
|-----------|-------|
| Human | 36,517 |
| A3C LSTM | 5,093 |
| A3C FF | 4,475 |
| Rainbow | 2,249 |
| DuDQN | 2,035 |
| DDQN | 1,219 |
| DQN | 697 |
| Random | 871 |

**Observation:** Asteroids is one of the games where DQN performs *below* the random agent on human starts.
The game is genuinely hard for standard deep RL.

---

## Source-by-Source Notes

---

### Playing Atari with Six Neurons
**Cuccu, Togelius, Cudre-Mauroux — AAMAS 2019**
[arxiv.org/abs/1806.01363](https://arxiv.org/abs/1806.01363)

The most directly relevant paper. The authors separate image processing from decision-making entirely: a
learned encoder compresses the raw game state into a handful of meaningful numbers, and then a tiny policy
network (6–18 neurons depending on the action space) operates on that compressed representation. They use
Increasing Dictionary Vector Quantization for the encoder and Exponential Natural Evolution Strategies to
train the policy. Results are "comparable and occasionally superior to state-of-the-art techniques using two
orders of magnitude more neurons."

**Key insight for us:** the policy network itself can be tiny — what matters is that the encoder gives it
something meaningful to work with. Our network receives 142 raw features and is expected to compress
*and* decide in a single hidden layer of 128 neurons with no bias. This paper suggests those two jobs
should be separated.

---

### Machine Learning Asteroids (Genetic Algorithm + NN)
**pjflanagan**
[github.com/pjflanagan/Machine-Learning-Asteroids](https://github.com/pjflanagan/Machine-Learning-Asteroids)

Ten ships compete simultaneously, evolved across generations using a genetic algorithm. The neural network
is very small: 8 inputs → 6 hidden → 4 outputs. The 8 inputs are *pre-computed distances to the nearest
asteroid in each of 8 directions* — the network never sees raw asteroid coordinates. Outputs are:
accelerate, fire, rotation-enable, rotation-direction. Bullets cost points if they miss, which discourages
random shooting. Ships are rewarded for moving.

**Key insight for us:** 8-6-4 is far smaller than our 142-128-6, yet it works because the
input preprocessing answers "is something nearby in this direction?" before the network sees anything.

---

### NEAT Asteroids with Raycasting
**Immodal**
[github.com/Immodal/asteroids](https://github.com/Immodal/asteroids) — [live demo](https://immodal.github.io/asteroids/)

A JavaScript implementation using NEAT (NeuroEvolution of Augmenting Topologies) with raycasting for
perception. NEAT starts from a minimal topology (just input→output connections) and grows the network by
mutation and speciation over generations — the architecture itself is evolved, not designed. Inputs come
from rays cast in multiple directions; outputs are the four actions. The live demo shows the network
structure and ray vectors simultaneously.

**Key insight for us:** NEAT removes the need to guess the right architecture entirely. The network grows
only as complex as it needs to be. Also uses raycasting — again, pre-computed geometry.

---

### A Neuroevolution Approach to General Atari Game Playing
**Hausknecht, Lehman, Miikkulainen, Stone — IEEE TCIAIG 2014**
[cs.utexas.edu/~mhauskn/papers/atari.pdf](https://www.cs.utexas.edu/~mhauskn/papers/atari.pdf)

A systematic comparison of four neuroevolution algorithms (conventional NE, CMA-ES, NEAT, HyperNEAT) across
three state representations (raw pixels, object features, noise) on 61 Atari games. Key findings:

- **NEAT achieved the highest mean z-score on object and noise representations**
- **HyperNEAT was the only algorithm that could handle raw pixels** — direct encoding fails completely
  on high-dimensional pixel inputs
- Evolved policies beat human high scores on Bowling, Kung Fu Master, and Video Pinball
- Found infinite score exploits in Gopher, Elevator Action, and Krull

**Key insight for us:** confirms that compact state representations strongly favour direct-encoding methods
(like NEAT or genetic algorithms). Raw pixels require indirect encoding (HyperNEAT) and scale poorly.
Our hand-crafted state is the right category — the question is whether 142 numbers is the right level of
pre-processing, or whether more geometry should be pre-computed.

---

### Developing an AI to Play Asteroids Part 1
**jrtechs blog**
[jrtechs.net/data-science/developing-an-ai-to-play-asteroids-part-1](https://jrtechs.net/data-science/developing-an-ai-to-play-asteroids-part-1)

Tried four approaches in order: random agent (~1000 pts), reflex agent (~2385 pts), genetic algorithm
(8090 on fixed seeds, ~2258 on random seeds), and deep Q-learning. The DQN architecture was the standard
DeepMind one: two conv layers + 256-unit FC + 14-class output. It trained for 48 hours across 950 games
and **converged to a suicidal policy** — consistently bad performance. The author traced two root causes:
(1) no separate target network, causing weights to shift every step; (2) no pixel-wise max across frames,
so asteroids disappeared from the observation entirely on alternating frames due to the Atari 2600 flickering.

**Key insight for us:** even with a large CNN, implementation bugs caused complete failure. Also: the
reflex agent (simple rules) outperformed the genetic algorithm on random seeds, suggesting that the
baseline for Asteroids is higher than it looks.

---

### AIsteroids (Q-Learning)
**datadaveshin**
[github.com/datadaveshin/AIsteroids](https://github.com/datadaveshin/AIsteroids)

Uses tabular Q-learning (not a neural network) on a Python Asteroids port. State is represented by
discrete game attributes — whether an asteroid is near, collision status — rather than raw coordinates.
No quantitative results reported. The project appears incomplete.

**Key insight for us:** even Q-learning on discretised features has been tried. The discrete feature
approach ("is there an asteroid nearby?") is conceptually similar to raycasting and avoids the raw
coordinate problem.

---

### AsteroidsDeepReinforcement (PPO)
**hdparks**
[github.com/hdparks/AsteroidsDeepReinforcement](https://github.com/hdparks/AsteroidsDeepReinforcement)

Uses PPO with a PyTorch PolicyNetwork trained on GPU. Describes three clear learning phases:

1. **Initial:** reckless, no strategy
2. **Middle:** discovered "Turn and Shoot" — the classic Asteroids maneuver
3. **Final:** mastered defensive movement, cleared asteroid waves

Trained in ~1,800 iterations (~6 hours on an average GPU). The agent eventually plays well. Also includes
a mode where humans can play against the trained agent.

**Key insight for us:** PPO discovers the "turn and shoot" behaviour naturally via reward shaping. It
finds shooting because score requires it — no need to handle the imbalanced action problem at all.
Reinforcement learning sidesteps behavioral cloning's class imbalance issue entirely.

---

### Exploring Reinforcement Learning: Training Agents to Play Asteroids
**Ekaterina Machneva — Medium / data-surge**
[medium.com/data-surge](https://medium.com/data-surge/exploring-reinforcement-learning-training-agents-to-play-asteroids-bfaa1b76fda9)
*(paywalled — 403 on fetch, summary from search results)*

Uses Stable Baselines3 with PPO and an MLP policy (not CNN) on a Gymnasium Asteroids environment. Trains
over a defined number of timesteps. A practical tutorial-style writeup showing it's straightforward to
get something working with modern RL libraries.

---

### Using Deep Learning Neural Networks to Play Asteroids
**HacWare — Medium**
[medium.com/@hacware](https://medium.com/@hacware/using-deep-learning-neural-networks-to-play-asteroids-part-1-cc770771077f)
*(paywalled — 403 on fetch, summary from search results)*

Uses a CNN-based approach with TensorFlow and raycasting for perception. Part 1 of a series. Combines
the raycasting compact-state idea with a neural network rather than neuroevolution.

---

### Asteroids-AI (Multiple Algorithms)
**lgoodridge**
[github.com/lgoodridge/Asteroids-AI](https://github.com/lgoodridge/Asteroids-AI)

A Python Asteroids implementation with a configurable experiment runner that supports multiple neural
network algorithms. Includes pre-trained "example brains," save/load of trained weights, reproducible
seeds, and sequential experiment comparison. Uses genetic evolution for training. No quantitative results
in the README but working example brains are included.

---

### OCAtari: Object-Centric Atari 2600 Environments
**Delfosse et al.**
[arxiv.org/abs/2306.08649](https://arxiv.org/abs/2306.08649)

Extends the Atari Learning Environment to provide structured object-centric states extracted from RAM,
covering all classic Atari games including Asteroids. Enables fair comparison between pixel-based and
object-based RL agents. Found that object-centric agents perform on par with or better than CNN agents
while running 50× faster. Provides asteroid position, size, colour; player position and heading; and
missile positions.

**Key insight for us:** this validates our choice of hand-crafted state over pixels. The paper also
shows what a well-specified object state looks like — they include object *size* as a feature, which
we omit.

---

### Towards Balanced Behavior Cloning from Imbalanced Datasets
**2025**
[arxiv.org/html/2508.06319v1](https://arxiv.org/html/2508.06319v1)

The only paper that directly addresses our class imbalance problem in behavioral cloning. Proves
mathematically that standard BC "leads to imbalanced learning — with policy parameters becoming weighted
averages favouring prevalent behaviours." The rare actions (in our case: SHOOT, ACCELERATE) are
systematically underweighted relative to common ones (TURN_LEFT, TURN_RIGHT).

Three proposed fixes:
1. **Equal weighting** — resample underrepresented actions to uniform frequency before training
2. **Relative loss weighting** — adjust per-class loss weights during training (what we tried)
3. **Meta-gradient** — automatically learn optimal reference loss per action class

Finds that loss-relative weighting (our approach) "depends critically on reference loss selection and
can fail with mixed optimal/suboptimal data." The meta-gradient approach outperforms both baselines.

**Key insight for us:** our class-weighting approach is theoretically sound but brittle — the right
weight magnitude is hard to choose, and we observed this directly when aggressive ACCELERATE weighting
(3.07) hurt survival. The meta-gradient approach would learn these weights automatically.

---

## What No One Seems to Have Done

- Behavioral cloning on a custom Asteroids implementation with a hand-crafted state vector (our exact setup)
- Pre-computing relative geometric features (asteroid angle relative to ship bearing, distance) as network
  inputs rather than raw absolute coordinates
- Combining a small geometric encoder with a tiny policy network in the Six Neurons style, but using
  supervised learning rather than neuroevolution

The closest analogues are the raycasting projects, which pre-compute "distance in direction θ" for discrete
angles. Our state could be improved by pre-computing equivalent features: for each asteroid, its angle
relative to the ship's current bearing, and its distance. This would give the network the geometric
relationships directly rather than requiring it to learn subtraction and dot products from 142 raw numbers.
