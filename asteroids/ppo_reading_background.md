# PPO Reading Background

Reading list for writing the PPO trainer by hand, with the goal of understanding what each part does rather than just getting a trained model. See `what_to_do_next.md` for why PPO and a critic are the next step, and `training_review.md` for the detailed analysis.

The current trainer (`training/policy_gradient.py`) is REINFORCE, which is where PPO starts, so most of the background is already in place.

## 1. Concepts: OpenAI Spinning Up

- **[Part 3: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).** Start here. It begins from REINFORCE, the algorithm `train_on_game_results` already uses. It then explains why subtracting a baseline helps and what an "advantage" is. That is the bridge from the current code to a critic.
- **[The PPO page](https://spinningup.openai.com/en/latest/algorithms/ppo.html).** Explains the clipped objective, which is the one new idea in PPO. Its short pseudocode is worth keeping open while writing the code.

## 2. The critic and GAE

The critic and GAE (Generalized Advantage Estimation, how the critic's predictions are turned into per-frame scores) are what fix the "dying is cheap" problem.

- **[Learning to Trust Your Critic: Grokking GAE](https://omegastick.github.io/2018/10/04/grokking-gae.html).** Intuition first, with a small worked example. Shows how the `lambda` setting slides between "trust the critic" and "trust the actual rewards".
- **[From Return to Advantage: Full Derivation to GAE](https://xinli95.github.io/AI-101/main/post_training/ppo_advantage.html).** The math, step by step from REINFORCE through baselines to GAE.

## 3. The practical details: "The 37 Implementation Details of PPO"

**[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)** (ICLR 2022 blog track) is the most useful single page for writing PPO yourself. The paper and textbook versions leave out small details that decide whether PPO works: how advantages are normalised, how a cut-off game is handled, how the learning rate is decayed, and so on. The post rebuilds PPO from scratch and explains each one.

Only the 13 "core" details matter here. Skip the Atari, continuous-action and LSTM sections.

## 4. Reference code to check against

**[CleanRL `ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)** is a single file of about 320 lines, written to match the 37-details post. It plays CartPole, which has a small set of discrete actions and an MLP network, the same setup as the polar2 model.

Write our version first, then read this to compare. Reading it first defeats the purpose.

## 5. The original paper (optional)

**[Schulman et al. 2017, "Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347).** Short and readable after the material above, but adds little that the earlier links do not.

## Suggested way to build it

Build in stages and check each one works before adding the next. That way each part is understood on its own, and it is clear which change caused a problem.

1. **Add a critic (no PPO yet).** Add a value network and use `return − predicted value` as the advantage. Remove the per-game normalisation. Log the critic's "explained variance", which says how well it predicts returns. If that stays near 0, the critic is not learning and nothing after this will work.
2. **Switch to GAE** with `gamma=0.999`. Check that the frames before deaths now get strongly negative advantages.
3. **Add PPO clipping and several passes over each batch.** This step gets more learning out of each batch of games.

Each stage lines up with a section of the Spinning Up and 37-details reading, so read one part, then implement it.

## Sources

- [Spinning Up — Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Learning to Trust Your Critic — Grokking GAE](https://omegastick.github.io/2018/10/04/grokking-gae.html)
- [From Return to Advantage: Full Derivation to GAE](https://xinli95.github.io/AI-101/main/post_training/ppo_advantage.html)
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- [CleanRL paper](https://arxiv.org/pdf/2111.08819) / [CleanRL PPO docs](https://cleanrl-q8bh8186d-vwxyzjn.vercel.app/rl-algorithms/ppo)
- [PPO paper (arXiv 1707.06347)](https://arxiv.org/abs/1707.06347)
