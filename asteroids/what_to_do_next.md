# What To Do Next

## Current status

Best model: `polar2_crisis_best.pth` — **146.4 avg over 1000 games** (all-time best). Produced by 100% crisis training for ~1300 iterations from `polar2_pg_wave5_exploit_best.pth` (130 avg). Already beats the heuristic AI (~139 avg).

## Reward normalization creates a "charge" incentive

The `discounted_rewards()` normalization (mean-subtract then divide by std) makes frames between kills have negative advantage. The model learns to minimize time between kills → charge at asteroids. This may not be optimal — keeping distance and shooting from range could score higher by avoiding deaths from close-range splits.

Crisis training collapses because it directly conflicts with this: crisis says "stay away from asteroids" while normalization says "get close fast." These are contradictory gradient signals on the same weights.

**Algorithm: clamp-to-zero normalization.** Standard normalization first (subtract mean, divide by std), then clamp negatives to zero: `ret = np.maximum(ret, 0)`. This keeps variance reduction from the normalization step but zeroes out all negative advantages. Kill frames get positive spikes, frames between kills get exactly zero — no gradient, no "stop doing that" signal. Visualized in `tools/visualize_reward_shaping.py` (bottom panel).

Note: an earlier idea (subtract min instead of clamping) didn't work — only one frame becomes zero and everything else shifts up by that amount, so valleys between kills still get small positive rewards instead of true zero.

**Optimal play style hypothesis:** Move away from asteroids, maintain maximum mean distance, shoot from range. Current normalization actively discourages this.

**Open question:** The current best model has thousands of iterations of charging behavior baked into its weights. Resuming from it with clamped normalization might not undo the charge — the model may need to be trained from scratch (or from an early checkpoint before charging was learned) to develop a distance-keeping play style.

## Crisis Training — All Mix Ratios Collapse

Tested with reward noise fix (skip zero-reward crisis games, no survival bonus for crisis): 100%, 97%, 90%, 50%, 10% crisis mix all show the same pattern — initial eval gain of +15-20 points in first 200-400 iterations, then steady collapse. The initial gain is real dodging skill transferring to normal gameplay. Then the crisis gradient overwhelms the shooting policy.

Burst + recovery (below) may still work but the normalization conflict is likely the deeper problem.

## Crisis Training Strategy: Burst + Recovery

Pure crisis training (100%) made massive improvements fast — best checkpoint at iter 1300 (146.4 avg) before catastrophic forgetting set in around iter 5000. The model didn't collapse until well after 1300 iterations.

**Idea: alternating burst training.**

1. Run 100% crisis training for 100–300 iterations (short burst, well within the safe window before collapse)
2. Switch to regular (non-crisis) training for some iterations to remind the model how to shoot and play normally
3. Repeat — each crisis burst teaches more dodging, each normal phase prevents forgetting

This avoids the frame-count dilution problem that killed mixed training (crisis games are ~100 frames vs ~5000 for normal games, so crisis signal gets drowned out in a combined batch). Separate bursts give each signal full gradient strength.

**Open questions:**
- How many normal-training iterations are needed to "recover" shooting behavior?
- Does the model retain dodging skills through the normal training phase, or does it forget those too?
- Could we detect the onset of collapse (e.g., eval score dropping) and auto-switch?
- What's the right burst length? 100 iters is conservative, 300 might extract more dodging skill per burst.

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

**Status:** Considering full approach. Selective variant is the preferred next step.

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

### Option 3b: Selective cross-entropy on dodge frames only — PREFERRED, DO THIS NEXT

The NN's shooting policy is already strong (~130 avg), possibly better than the heuristic's. What's missing is dodge behavior — the NN doesn't know what to do when a collision is imminent. Full cross-entropy would overwrite the good shooting policy and hit the class imbalance problem.

**Proposal:** Filter existing recorded heuristic games to only frames where a collision is imminent (using the same forward-projection check the heuristic uses, lines 322-365 of heuristic.py). Cross-entropy train on just those frames to teach the NN dodge behavior without touching its shooting policy. Then PG fine-tune.

**Implementation:**
1. Use existing recorded game data (raw geometry format)
2. For each frame, run the heuristic's collision prediction (project player + asteroid positions forward 49 ticks, check geometric overlap)
3. Keep only frames where collision is imminent — label with the heuristic's action
4. Cross-entropy train the current best polar2 model on these filtered frames
5. PG fine-tune from the resulting checkpoint

**Why this should work:**
- Directly teaches the one behavior the NN is missing (dodging)
- Avoids class imbalance — dodge frames are dominated by thrust/turn, not the 93% turning of the full dataset
- Doesn't overwrite existing shooting policy since non-dodge frames are excluded
- Polar2 features already include TTI, so the NN has the inputs to learn when and how to dodge — it just hasn't seen enough dodge situations during PG training
- No need to re-record data — apply the collision check to existing recordings

## Option 4: Curriculum training on later waves — DONE, plateaued at ~131

Start some or all training games at higher waves so the agent sees fast
asteroids more often. Most training episodes end in early waves, so the bulk
of gradient updates come from slow-asteroid conditions. The agent adapts to
wave 1-3 physics then hits a wall when later waves demand tighter margins.

**Training distribution mismatch:**

Speed multiplier is `1.1^(wave-1)` — wave 1 starts at 1.0, each subsequent wave multiplies by 1.1:

| Wave | Multiplier | Asteroid speed | Time to cross 50px |
|------|-----------|---------------|-------------------|
| 1 | 1.0 | 100 px/s | 30 frames |
| 3 | 1.21 | 121 px/s | 25 frames |
| 5 | 1.46 | 146 px/s | 21 frames |
| 7 | 1.77 | 177 px/s | 17 frames |

Current NN avg ~118 points = partway through wave 5 (108 points to clear wave 4). Heuristic avg ~139 = partway through wave 6. The gap is roughly one wave of survival.

Orthogonal to reward shaping — addresses *how often* hard situations appear, not *what to learn from them*. Combine with death penalty (-0.1) and entropy (0.005).

**Implementation plan:**

1. `Game.__init__(width, height, starting_wave=1)` — set `asteroid_speed_multiplier = 1.1^(starting_wave-1)` so initial asteroids spawn at the correct speed. All existing callers default to wave 1, no changes needed.
2. `training/policy_gradient.py` — add `--starting-wave` CLI arg, thread through worker args tuple to `Game(width, height, starting_wave)`.
3. Start training at wave 5 — resume from `polar2_pg_ent005_best.pth`, keep death penalty (-0.1 over 60 frames) and entropy (0.005), reset max_score to 0 since scores are not comparable across different starting waves.

**Results:** 1000-game wave-1 benchmarks: avg 128 at 12k, avg 131 at 18k (peak), regressed to 128 at 24k. Follow-up exploit run (entropy=0) from 18k checkpoint: cleaner actions but same ~130 avg after 39.5k iters. Plateaued. Still improving despite flat wave-5 batch averages.

## Option 5: Reduce or remove entropy bonus

**Status:** Not started. Try after current wave-5 run plateaus.

Visual observation shows the policy rapidly alternating between accelerate and decelerate on consecutive frames, with turns and shots intermixed. This accel/decel flicker is effectively a no-op (net zero displacement) and wastes frames that could be spent on useful actions. The entropy bonus (0.005) is the likely cause — it pressures the policy to keep all 6 actions in use, so it fills gaps with meaningless thrust toggles rather than committing to a direction or staying still.

**Proposal:** Continue from the best wave-5 checkpoint with entropy reduced to 0 or 0.001. The policy has already developed a broad action distribution from the entropy-guided training; removing the bonus should let it tighten up and stop wasting frames.

**Risk:** Without entropy the policy could collapse back to a narrow action set (the `polar2_pg_exploit` failure — avg dropped to 50-60). Mitigated by starting from a much stronger checkpoint (131 avg vs 70 at the time). Could try 0.001 as a middle ground.
