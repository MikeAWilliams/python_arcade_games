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

## Training distribution mismatch at higher waves

The speed multiplier is 1.1x per wave, so asteroids get meaningfully faster:

| Wave | Multiplier | Asteroid speed | Time to cross 50px |
|------|-----------|---------------|-------------------|
| 1 | 1.0 | 100 px/s | 30 frames |
| 3 | 1.33 | 133 px/s | 22 frames |
| 5 | 1.61 | 161 px/s | 19 frames |
| 7 | 1.95 | 195 px/s | 15 frames |

Most training episodes end in early waves, so the bulk of gradient updates
come from slow-asteroid conditions. The agent gets well-adapted to wave 1-3
physics, then hits a wall when later waves demand tighter evasion margins.
A charging strategy that works at 100 px/s becomes suicidal at 161 px/s.

The death penalty helps indirectly — most deaths probably happen at the
transition to faster waves, which is exactly where the agent needs the
strongest learning signal.

## Charging is probably a local optimum

The charge strategy maximizes short-term kill rate but is a dead end — the
evasion physics above show you physically can't dodge splits at close range.
Bullet speed (500 px/s) is 3-5x asteroid speed, so there's no accuracy
benefit to closing distance. The optimal strategy is likely kiting: maintain
maximum distance, use the whole screen, shoot accurately from range. The
heuristic AI already plays something close to this.

The death penalty should create pressure to discover that the high-scoring
strategy is also the safe one: kill from range, keep moving, use the space.

## Resume command for polar2_pg_ent005

Run was stopped at iteration ~14.5k of 60k. Still improving (best 147.76).
Resume with:

```bash
python training/policy_gradient.py --model-type polar2 --run-name polar2_pg_ent005 --entropy-coeff 0.005 --checkpoint nn_checkpoints/polar2_pg_ent005_best.pth
```

Note: this restarts from epoch 0 but with the best weights and optimizer
state. The run will overwrite the log file. The best score (147.76) is
preserved in the checkpoint so the max_score tracker picks up where it
left off.

## Orthogonal idea: curriculum training on later waves

Start some or all training games at higher waves so the agent sees fast
asteroids more often. This directly addresses the distribution mismatch
without changing the reward structure. Could be combined with the death
penalty — they solve different problems (what to learn from death vs.
how often the agent encounters hard conditions).
