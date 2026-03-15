# Entropy Bonus for Policy Gradient Exploration

The polar NN model has converged on a "stand and shoot" local optimum. It aims (a big win) but doesn't dodge or lead targets. The state vector (distance, closing speed, relative angle) contains enough information for dodging, and the reward structure supports it (surviving = more future kill rewards), but the model never explores evasive actions enough to discover this.

## Idea: Add entropy bonus to the policy gradient loss

**How it works:** Currently the loss is `-mean(log_prob * advantage)`. Add a term: `-entropy_coeff * mean(entropy)` where entropy = `-sum(p * log(p))` over the action distribution. This penalizes the model for being too certain about its actions, forcing it to maintain some randomness in its policy. A typical coefficient is 0.01-0.05.

**Implementation:** In `train_on_game_results()` in `training/policy_gradient.py`, after computing log_probs:
1. Compute `probs = F.softmax(logits, dim=1)`
2. Compute `entropy = -torch.sum(probs * log_probs, dim=1)`
3. Modify loss: `loss = -torch.mean(log_probs * advantages) - entropy_coeff * torch.mean(entropy)`

## What it might accomplish

By keeping the policy stochastic, the model would occasionally take random evasive actions during training episodes. If dodging an incoming asteroid leads to a longer game with more kills, that episode gets a strong reward signal, and the gradient update reinforces the dodging behavior. Without entropy, the policy becomes deterministic too quickly and never stumbles into these discoveries.

## Why

The model is stuck in a local optimum. Standard REINFORCE has no built-in exploration mechanism once the policy sharpens. Entropy regularization is the standard fix for this in policy gradient methods (used in A2C, PPO, etc.).

## How to apply

Try this after the current training run plateaus. Start with entropy_coeff=0.01 and increase if the policy is still too deterministic. Can be added as a CLI arg.
