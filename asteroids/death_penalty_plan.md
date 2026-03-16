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

## Evasion physics (justifying the 60-frame window)

Constants at wave 3 (speed multiplier 1.1^3 = 1.331):
- Player acceleration: 180 px/s²
- Player radius: 20 px
- Small asteroid radius: 30 px (the splits from a medium)
- Collision distance (player + small): 50 px
- Asteroid speed: 100 × 1.331 = 133 px/s

Ship lateral displacement from rest under constant acceleration (distance = ½at²):

| Frames | Time (s) | Distance moved | Notes |
|--------|----------|---------------|-------|
| 10 | 0.167 | 2.5 px | Barely moved |
| 20 | 0.333 | 10 px | Still within collision range |
| 30 | 0.500 | 22.5 px | Just clearing its own radius |
| 36 | 0.600 | 32.4 px | Clears half the collision distance |
| 45 | 0.750 | 50.6 px | Just clears collision distance |
| 60 | 1.000 | 90 px | Comfortable escape |

A small asteroid at 133 px/s covers 50 px in 0.375s (22 frames). So if a split
happens at close range, the asteroid reaches you in ~22 frames but you need ~45
frames to dodge it. Close-range splits are physically unavoidable.

Turn rate is π rad/s, so rotating 90° takes 30 frames and 180° takes 60.
But the ship can thrust forward and backward, so the worst-case rotation to
an escape vector is 90° (30 frames) — not 180°.

Worst-case evasion from a close-range split:
- 30 frames to turn 90° to the escape direction
- 45 frames to accelerate one collision distance
- **~60-75 frames total**

The decisions that matter — whether to charge in that close — happen before
the split. 60 frames is a reasonable penalty window that covers the physical
evasion time and catches the approach decision.

## Design considerations
- Decay ramps 0→1 so the last frame gets full penalty, actions further back get less blame
- Magnitude (-0.5) must be less than +1 kill reward so "charge, kill, die" still nets positive — but "charge, kill, survive" is clearly better
- N=60 frames (~1 second) covers the reaction window after a split
- The hope is the agent learns its own implicit safe distance rather than needing a hardcoded threshold
- Try this after evaluating the `polar2_pg_ent005` run results
