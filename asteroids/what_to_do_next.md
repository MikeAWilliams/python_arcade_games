# What To Do Next

Options and ideas for improving the NN AI beyond its current ~118 avg (100-game benchmark) to beat the heuristic AI (~139 avg).

## Current status

Best run: `polar2_pg_ent005` (best batch avg 159, 100-game benchmark avg ~118). The "best" metric tracks peak batch-of-32 averages across thousands of batches, so it overstates true performance due to statistical variance. The 100-game benchmark is the real measure.

Gap to close: ~21 points vs heuristic (118 vs 139).

## Option 1: Death penalty reward shaping (DONE — limited effect)

**Implemented** in `training/policy_gradient.py` via `--death-penalty` and `--death-penalty-frames` flags.

Add a decaying negative reward to the last N frames before death to give the agent a direct learning signal about what killed it.

**Results so far:**
- `-0.5` over 60 frames (`ent005_dp`): No improvement at all over 15k iterations. Cumulative penalty (~-15) overwhelmed the +1 kill rewards, drowning out the scoring signal.
- `-0.1` over 60 frames (`ent005_dp01`): Marginal improvement — two new bests (160, 163) after 12k iterations. Some subtle dodging observed visually. But 100-game benchmark didn't improve meaningfully.

**Why it may not work:** The per-game normalization in `discounted_rewards` already makes end-of-life frames the lowest-advantage frames. Adding explicit penalty makes them *more* negative but doesn't fundamentally change the gradient direction. The model was already being told "your last frames were your worst."

### Evasion physics (justifying the 60-frame window)

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

### Design notes
- Decay ramps 0→1 so the last frame gets full penalty, actions further back get less blame
- Magnitude must be less than +1 kill reward so "charge, kill, die" still nets positive
- N=60 frames (~1 second) covers the reaction window after a split
- Charging is probably a local optimum — bullet speed (500 px/s) is 3-5x asteroid speed, so there's no accuracy benefit to closing distance

## Option 2: Add collision prediction feature to polar2

**Status:** Not started. Strongest theoretical case for closing the gap.

The heuristic's evasion advantage comes from explicit tick-by-tick collision prediction (lines 322-365 of heuristic.py). It projects both player and asteroid positions forward for 49 ticks and checks geometric overlap. This catches oblique approaches — asteroids crossing the player's path from the side.

Polar2's time-to-impact (TTI) only captures the *radial* component of closing speed. An asteroid approaching at an oblique angle has low radial closing speed, so TTI reads as safe, but the heuristic's projection catches it as a collision. These oblique hits may be a significant fraction of deaths.

**Proposal:** Add a `min_collision_ticks` feature — do the same forward projection the heuristic does (or approximation) and give the NN a single scalar: "nearest collision in N ticks, or max if none." This directly encodes the information the heuristic uses for its mode switch without the NN needing to learn it from raw geometry.

**Why this might work:** The heuristic's entire dodge-vs-shoot decision rests on this collision prediction. It has a hard binary mode switch: if *any* asteroid will collide within 49 ticks, it completely stops shooting and fully commits to evasion. Giving the NN this same signal removes the hardest part of what it needs to learn — it no longer has to discover the concept of "imminent collision" from TTI and angles, it gets told directly.

**Risk:** Could be seen as "cheating" — giving the NN the heuristic's answer rather than letting it learn. But the goal is to beat the heuristic, and the NN still needs to learn what *action* to take given this signal.

## Option 3: Cross-entropy training from heuristic data

**Status:** Considering.

Train the NN via supervised learning on recorded heuristic AI gameplay data. The cross-entropy pipeline already exists in `training/cross_entropy.py`.

Training pipeline already exists in `training/cross_entropy.py`. Heuristic gameplay data already recorded in raw geometry format — would need to convert to polar2 features or adapt cross_entropy.py to compute polar2 state from raw recordings.

**Pros:**
- Direct signal for what actions to take in every state, including dodge states
- Would teach the mode switch behavior (shoot vs dodge) explicitly
- Fast to train — no game simulation needed, just batch gradient descent on recorded data
- Could serve as initialization for further PG fine-tuning

**Cons:**
- Ceiling is heuristic performance — can't learn better than the teacher
- Previous cross-entropy attempt (with raw geometry model) failed due to catastrophic class imbalance (93% turning actions) — see IO_SUMMARY.md for analysis
- Would need to fix the class imbalance problem first (class-weighted loss, smaller batches, etc.)

**Hybrid approach:** Cross-entropy to get close to heuristic performance (especially learning the dodge behavior), then PG fine-tuning to push beyond it. This avoids starting PG from scratch without dodge knowledge.

## Option 4: Curriculum training on later waves

Start some or all training games at higher waves so the agent sees fast
asteroids more often. Most training episodes end in early waves, so the bulk
of gradient updates come from slow-asteroid conditions. The agent adapts to
wave 1-3 physics then hits a wall when later waves demand tighter margins.

**Training distribution mismatch:**

| Wave | Multiplier | Asteroid speed | Time to cross 50px |
|------|-----------|---------------|-------------------|
| 1 | 1.0 | 100 px/s | 30 frames |
| 3 | 1.33 | 133 px/s | 22 frames |
| 5 | 1.61 | 161 px/s | 19 frames |
| 7 | 1.95 | 195 px/s | 15 frames |

Orthogonal to reward shaping — addresses *how often* hard situations appear, not *what to learn from them*. Could be combined with other approaches.
