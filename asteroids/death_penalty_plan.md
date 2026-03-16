# Death Penalty Reward Shaping

Add a death penalty to the policy gradient reward signal to give the agent a direct learning signal about what killed it.

**Why:** With the current reward structure (only +1 per kill, +0.001 survival bonus), dying produces near-zero return for the final actions — the agent gets almost no signal about what killed it. The `polar2_pg_exploit` run showed the agent learned to charge at asteroids (which increases kill rate) but dies to splits at close range because there's no penalty for the actions that led to death.

**How to apply:** In `training/policy_gradient.py`, insert between lines 189 and 191 (after survival bonus, before discounted rewards computation):

```python
# Death penalty: penalize last N frames before death
death_penalty_frames = 60  # ~1 second
death_penalty = -0.5       # tunable
for i in range(max(0, len(rewards) - death_penalty_frames), len(rewards)):
    decay = (i - (len(rewards) - death_penalty_frames)) / death_penalty_frames  # 0→1
    rewards[i] += death_penalty * decay
```

## Design considerations
- Decay ramps 0→1 so the last frame gets full penalty, actions further back get less blame
- Magnitude (-0.5) must be less than +1 kill reward so "charge, kill, die" still nets positive — but "charge, kill, survive" is clearly better
- N=60 frames (~1 second) covers the reaction window after a split
- The hope is the agent learns its own implicit safe distance rather than needing a hardcoded threshold
- Try this after evaluating the `polar2_pg_ent005` run results
