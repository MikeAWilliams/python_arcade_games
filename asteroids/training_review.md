# Training Review (2026-11-13)

A review of the notes in this folder against the code in `training/policy_gradient.py`,
`asteroids/ai/polar2_nn.py` and `asteroids/core/game.py`, plus a few measurements on the
saved checkpoints. It supersedes the diagnosis in `what_to_do_next.md` where the two disagree.

**Short version.** The plateau is not a capacity problem and mostly not a feature problem.
The training objective does not contain the thing you want. With `gamma=0.99` at 60 fps and
per-episode return normalization, dying costs the agent about 2.5 kills, so it has been
trained to maximise kill rate over the next ~1.7 seconds, and it does that well. On top of
that, the "best" numbers used to steer the experiments are mostly noise, so several
conclusions in the notes (death penalty helps a bit, crisis gives +15–20 before collapsing,
close-range splits are the killer) do not hold up.

---

## 1. What was measured

All numbers below are from the saved checkpoints, sampled policy (as `main_headless.py`
plays it), 1280×720, default heuristic parameters.

### 1.1 Re-benchmarks of "best" checkpoints

| Checkpoint | Score recorded in checkpoint / notes | Re-benchmark (mean ± SEM) | Games |
|---|---|---|---|
| `polar2_pg_ent005_best.pth` | 159.0 ("best") | **107.6 ± 1.9** | 960 |
| `polar2_pg_ent005_dp01_best.pth` | 163.6 ("best") | **115.4 ± 2.0** | 960 |
| `polar2_pg_wave5_exploit_best.pth` | ~130 (1000-game) | **134.1 ± 2.1** | 960 |
| `polar2_crisis_best.pth` | 146.4 (1000-game) | **150.4 ± 3.0** | 480 |
| Heuristic | ~139 | **137.9 ± 2.5** | 480 |

Per-game score standard deviation is **~65** for every policy. So:

- A 32-game training batch average has a standard error of ~11.5 points.
- A 100-game eval has a standard error of ~6.5 points.
- The maximum of a few hundred such evals sits 15–20 points (100-game) or 30–50 points
  (32-game batch) above the true mean even if the policy never changes.

The 159 and 163 "bests" were maxima over 32-game batches and overstate the true score by
about 50 points. Only the 1000-game benchmarks in the notes are trustworthy, and they hold up.

### 1.2 How the best model dies (`polar2_crisis_best.pth`, 480 games)

| Measurement | NN (best) | Heuristic |
|---|---|---|
| Mean score | 150.4 | 137.9 |
| Mean frames survived | 6118 | 5805 |
| Kills per frame | 0.0246 | 0.0238 |
| Killer asteroid was created < 60 frames before the hit (fresh split) | **4%** | 4% |
| Death within 60 / 120 frames of a wave start | 9% / 22% | 5% / 13% |
| Player within 60 px of a wall at death | 0.6% | 8% |
| Killer bounced off a wall in the previous 90 frames | 52% | 50% |
| Same statistic for all live asteroids (baseline) | 49% | 48% |
| Killer size: small / medium / big | 38% / 23% / 33% | 37% / 28% / 35% |
| Mean player speed at death | 45 px/s | 87 px/s |
| Action mix (L / R / accel / decel / shoot / none) | 33 / 33 / 17 / 11 / 5.5 / 0.2 % | 46.5 / 46.8 / 0.8 / 1.8 / 4.1 / 0 % |

"Warning time" = number of consecutive frames before death during which a straight-line
projection of the current player and asteroid velocities already predicted the hit
(the same test the heuristic uses, without its 49-tick limit):

| Warning time before death | NN (best) | Heuristic |
|---|---|---|
| < 22 frames | 13% | 18% |
| < 45 frames | 27% | 43% |
| 45–90 frames | 42% | 27% |
| ≥ 90 frames (1.5 s) | 31% | 30% |
| Median | 66 frames | 54 frames |

What the NN was doing during that warning window (last ≤ 90 frames of it):

| Warning time | Thrust (accel + decel) | Turning | Shooting |
|---|---|---|---|
| < 45 frames | 34% | 62% | 3.6% |
| 45–90 frames | **12%** | **85%** | 3.0% |
| ≥ 90 frames | **13%** | **83%** | 3.7% |

The heuristic thrusts 43–70% of the time in the same windows.

Also checked: 45 frames before death the killer's time-to-impact feature was under 2 s in
98% of deaths, and the killer was always inside the 9 asteroid slots the network sees.

### 1.3 What this says

1. **73% of deaths come with ≥ 0.75 s of straight-line warning, and 31% with ≥ 1.5 s.**
   The threat is visible in the existing polar2 inputs. The policy keeps turning to aim and
   barely thrusts. This is a policy that was never taught that dying is expensive, not a
   policy that cannot see.
2. **Close-range splits are not the problem.** Only 4% of deaths are from an asteroid
   created in the previous second. The "splits are physically unavoidable" section of
   `what_to_do_next.md` is correct physics but describes a rare event.
3. **Wall bounces are not a disproportionate killer** (52% vs a 49% baseline). Wall
   information is missing from the state, but it is not what is holding the score down.
4. **Wave starts are over-represented**: the first 2 seconds of a wave are ~11% of play
   time but 22% of NN deaths (13% for the heuristic). The heuristic shows roughly half of
   these are avoidable. The rest are spawn geometry: a big asteroid can spawn with an
   80 px edge gap (190 px centre distance) heading straight at a stationary player, and at
   wave 6+ that is under 0.5 s. This is a noise floor for every agent.
5. Greedy (argmax) play of the same model scores 140.1 ± 3.0, about 10 below sampled play.
   The policy leans on its own randomness; a well-trained policy should not.

---

## 2. Errors found

### 2.1 The objective barely penalises death (main error)

`discounted_rewards()` uses `gamma=0.99` per frame at 60 fps. The effective horizon is
`1 / (1 - gamma) = 100 frames = 1.67 s`.

- Kill rate is ~0.0246 per frame. The discounted value of staying alive forever is
  `0.0246 / 0.01 ≈ 2.5` kills. That is the entire cost of dying as the trainer sees it.
- The true cost of dying is all future score. With a roughly constant hazard that is on the
  order of 100+ points at any moment.
- So getting one kill a fraction of a second sooner is worth as much as a large reduction
  in death risk. Charging, sitting still and never interrupting aiming to dodge are the
  correct solutions to the objective as written.

Per-episode normalization (`(ret - mean) / std` inside each game) makes it worse:

- Every game is rescaled to zero mean and unit variance, so a 300-point game and a
  20-point game push on the weights equally hard. Whatever information existed about
  "this game went better than that one" is deleted.
- What remains is purely within-game timing: frames shortly before a kill are positive,
  everything else negative.

`what_to_do_next.md` attributes the charge incentive to normalization making between-kill
frames negative. That is a symptom. The cause is the 1.67 s horizon. Any baseline makes
some frames negative; that is how a baseline works.

**The clamp-to-zero idea should not be tried.** Clamping negative advantages to zero:

- removes the only "don't do that" signal, including the frames before death, which are
  the lowest-return frames in the game;
- makes the gradient estimate biased (a baseline may depend on state, never on the return);
- leaves the horizon unchanged, so pre-kill frames are still the only ones reinforced.

It would produce the same charging policy with less dodging signal than now.

### 2.2 Model selection and experiment conclusions are driven by noise

See 1.1. Consequences already visible in the notes:

- `polar2_crisis_mix10`: "best eval 144.95 at iter 0". Iteration 0 is the unmodified
  starting checkpoint, which is a 134-average model. The +11 is sampling noise.
- All crisis-mix runs report a best eval in the first 0–400 iterations of 133–155 starting
  from the same 134 model. That is the expected maximum of a handful of 100-game evals.
  It is not evidence of "real dodging skill transferring". (The later collapse to 60–90
  is real; that is many standard errors.)
- Death penalty −0.1 "two new bests (160, 163)": true score 115 vs 108 for its parent.
  A small real gain, far smaller than the bests suggest.
- `polar2_crisis_best` (iter 1300 of 100% crisis) is a real gain: 134 → 150.

Rule of thumb: with σ ≈ 65, telling apart two policies 5 points apart needs ~1400 games
each with independent seeds, or a few hundred with paired seeds.

### 2.3 Why every crisis mix collapses the same way

The notes blame a conflict between "stay away" and "charge". The mechanics of the code give
a simpler explanation:

1. Crisis and normal games get **separate `optimizer.step()` calls on the same Adam
   optimizer**. Adam normalises step size by gradient history, so each step moves the
   weights by roughly `lr` regardless of how much data is behind it.
2. A normal step averages ~190,000 frames. A crisis step at 10% mix averages a few hundred
   frames (3 games × ~100 frames, minus the games that were skipped). Its gradient is
   almost pure noise, has a much larger norm (clipped to 1.0), and gets a full-size step.
3. The large crisis gradients also dominate Adam's shared second-moment estimate, which
   shrinks the normal-game steps. The effective mix is close to 100% crisis at any ratio.
   That is why 10%, 50%, 90% and 100% all look alike.
4. Failed crisis games are discarded, so crisis training only ever sees successes. The
   deaths, which are the dodging lesson, are thrown away. What gets reinforced is "the
   frames just before the wave-clear +27", i.e. shooting the last asteroid.
5. Crisis scoring (no per-kill reward, +27 on clear) is a different reward function from
   the real game, trained into the same weights.

The one real gain (iter 1300 of the 100% run) shows hard-situation practice has value. The
delivery mechanism is what breaks. Section 4.4 has a replacement.

### 2.4 State vector warts (polar2)

None of these explains the plateau by itself (see 1.3), but they cost learning speed and should
be fixed together when the input layout next changes.

1. **Empty slots look like the worst possible threat.** Padding is `[0, 0, 0, 0]`, which
   decodes as "edge distance 0, dead ahead, impact now". A far, safe asteroid is
   `[~1, x, 1, x]`. The network has to learn that the most dangerous-looking pattern
   means "nothing here". Pad with `distance=1, tti=1` and add a presence flag.
2. **TTI resolution is in the wrong place.** `tti / 10` clamped at 10 s puts the whole
   decision range (0–1 s) into 0–0.1. Use a 3 s clamp, or an urgency feature `1 / (1 + tti)`.
3. **TTI is radial only.** It ignores whether the asteroid will actually hit. Replace or
   supplement with closest point of approach, which is analytic and needs no tick loop:
   `t_cpa = -(r·v) / |v|²` (clamped to ≥ 0) and `miss = |r + v·t_cpa| - (R_player + R_asteroid)`,
   with `r`, `v` the relative position and velocity. This is Option 2 of
   `what_to_do_next.md` in closed form.
4. **Relative angle wraps at ±1 directly behind the ship.** That is exactly where the
   "threat behind → accelerate" decision lives. Encode as `sin` and `cos`.
5. **Asteroid radius was removed.** It determines the angular width of the target and what
   a hit will spawn. A third of deaths are to big asteroids. Put it back as one input.
6. Missing but lower priority: wave speed multiplier (lets a critic predict value),
   distance to walls along and across the bearing.

### 2.5 Smaller issues

- **One Adam step per ~190k frames.** Each batch of 32 games is used for a single gradient
  step at `lr=1e-4`. Sample efficiency is very poor; PPO-style reuse (several epochs of
  minibatches per batch) gets 10–50× more learning per simulated frame.
- **60 decisions per second from a sampling policy.** Independent sampling every 1/60 s is
  what produces the accel/decel flicker; the entropy bonus only makes it more visible.
  It also makes episodes ~6000 decisions long, which is what makes credit assignment hard.
- **`best_training.pth`** is chosen on a 32-game average. Ignore it.
- **Selective cross-entropy on heuristic dodge frames (Option 3b)** should be demoted.
  The heuristic scores lower than the current NN and 57% of its own deaths had ≥ 45 frames
  of warning while it was thrusting. It is not a good dodge teacher.
- `polar_model_improvements.md` says the combined change gives 31 inputs. The implemented
  polar2 has 40 (4 global + 4 × 9).
- `CLAUDE.md` does not list polar2, the crisis flags or `--starting-wave`.

---

## 3. What was not wrong

- Network size. 40 → 128 → 64 → 6 is plenty for this state.
- The polar feature idea. It is what took scores from ~10 to ~130.
- Entropy bonus as a way out of the two-action collapse.
- Wave-5 curriculum. It gave a real, measured gain (108 → 134).
- Simulation throughput. About 245k frames/s on 16 cores including inference; compute is
  not the constraint.

---

## 4. Recommended plan

In order. Step 1 is a prerequisite for believing anything after it.

### 4.1 Fix evaluation first (small change)

- Select and report on **1000 games with fixed seeds** (`random.seed(base + game_id)`, as
  `GameRunner` already does). Fixed seeds make comparisons between checkpoints paired,
  which removes most of the spawn-luck variance.
- During training, evaluate less often but properly (e.g. 500 fixed-seed games every 500
  iterations) and save "best" only on that.
- Log the true mean with its standard error. Never use a batch maximum as a result.
- Log both sampled and greedy scores. The gap is a health indicator.
- Add two diagnostics from section 1.2 to the eval: fraction of deaths with ≥ 45 frames of
  warning, and thrust fraction inside the warning window. These move before the score does.

### 4.2 Fix the objective: actor-critic with a long horizon (the main change)

Replace REINFORCE + per-episode normalization with PPO (or A2C) + a learned value function
+ GAE. Concretely, in `training/policy_gradient.py`:

- **Reward:** +1 per kill. Nothing else. Remove the survival bonus and death penalty.
- **Discount:** `gamma = 0.999` per frame (horizon 1000 frames ≈ 17 s). Value of being alive
  becomes ~25 kills, so a death is a −25 event against +1 for a kill.
- **Critic:** a separate MLP `40 → 128 → 64 → 1`. Keeping it separate means existing actor
  checkpoints still load. Terminal state (death) has value 0. No bootstrapping on death.
- **Advantages:** GAE with `lambda = 0.95`, normalised over the whole batch, never per game.
- **Why this delivers the death signal the death-penalty hack could not:** just before a
  death the critic still predicts ~25; the outcome is 0. That TD error lands on the frames
  where the situation went from fine to doomed, scaled automatically, and GAE spreads it
  back ~20 frames with decaying weight. The manual penalty put a fixed ramp on the last 60
  frames whether or not anything could be done in them, then per-episode normalization
  rescaled it.
- **Rollouts:** keep the existing worker pool, but return `(state, action, reward, done,
  log_prob)` per frame and compute values and GAE in the learner. Fixed-length segments
  (e.g. 4096 frames per worker, environment carried over between iterations, bootstrap
  from `V` at the cut) work better than whole games with a long horizon.
- **Update:** PPO clip 0.2, 4 epochs, minibatch 4096, `lr 3e-4`, value coefficient 0.5,
  gradient clip 0.5, entropy 0.01 decayed to 0.001. Scale value targets (divide reward by
  10 or normalise returns) so the critic loss is O(1).
- **Warm start:** load `polar2_crisis_best.pth` into the actor, freeze it, and train only
  the critic for a few hundred iterations until explained variance stops rising. Then
  unfreeze. Starting the actor update with an untrained critic will wreck the policy.
- **Also run one from scratch.** The current weights have turret behaviour baked in and
  may be a poor starting point (the open question in `what_to_do_next.md`). With PPO's
  sample reuse a from-scratch run is affordable.

What to expect if it works: thrust fraction inside warning windows rises well above 12%,
warned deaths fall, mean frames survived rises faster than kills per frame, greedy and
sampled scores converge.

A cheap sanity check before the full rewrite: critic explained variance. If the critic
cannot predict returns at `gamma=0.999` (explained variance stuck near 0), the state is
missing something the value depends on (wave speed is the first suspect) and 2.4 moves up.

### 4.3 Shorten the decision horizon (try alongside 4.2)

Action repeat of 3 frames (decide at 20 Hz), with `gamma = 0.999³ ≈ 0.997` per decision.

- Episodes drop from ~6000 to ~2000 decisions.
- Exploration becomes coherent: a sampled "accelerate" lasts 50 ms and actually moves the ship.
- Turn granularity becomes 9° per decision. The GA-tuned heuristic shoots at a 24°
  tolerance (0.425 rad), so this is not too coarse.

This changes the agent's interface, so benchmark the heuristic under the same repeat for a
fair comparison, or treat it as a training-only setting and evaluate at repeat 1.

### 4.4 Replace crisis mode with restarts from the agent's own near-death states

Keep the idea, change the delivery:

- During normal rollouts keep a ring buffer of `copy.deepcopy(game)` snapshots every ~30
  frames. When the agent dies, push the snapshots from 2–4 s before death into a pool.
- Start some fraction of episodes (say 25%) from a pool snapshot instead of a fresh game.
- Same reward, same scoring, same optimizer step, same batch as normal play. No separate
  gradient update, no discarded failures, no synthetic geometry.

These are on-distribution hard states, and the agent sees both outcomes of each. With a
critic they also cannot dominate the gradient just by being short.

### 4.5 polar3 state

Apply 2.4 items 1–5 (and wave speed multiplier) as a new `polar3` model type and train it
with the 4.2 trainer. Do this after 4.2 shows movement on polar2, so the effect of the
objective change and the feature change can be told apart.

### 4.6 Alternative that sidesteps credit assignment: evolution strategies

The benchmark metric is total score per game; ES optimises exactly that with no discount,
no critic and no per-frame credit assignment. The policy has ~14k parameters and the
simulator does ~245k frames/s, which is within range.

- Start from `polar2_crisis_best.pth`, play greedy (argmax) so evaluation is deterministic
  given the seed.
- OpenAI-ES with antithetic pairs: population 128–256, noise σ ≈ 0.02, each member plays
  the **same** 8–16 seeds within a generation (new seeds each generation), rank-normalised
  fitness, Adam on the ES gradient.
- Cost is roughly 256 × 8 games × 6000 frames ≈ 12M frames ≈ 1 minute per generation.
- `training/genetic.py` already has the parallel evaluation plumbing.

It is slower per unit of improvement than a working PPO, but it cannot be fooled by the
objective, so it is a useful cross-check: if ES climbs from 150 and PPO does not, the PPO
setup is still wrong.

### 4.7 Things to stop doing

- Clamp-to-zero normalization (2.1).
- Burst/recovery crisis schedules and further crisis-mix ratios (2.3).
- Hand-tuned death penalties and survival bonuses (the critic replaces them).
- Selective cross-entropy from the heuristic (2.5).
- Reading anything into a single 100-game eval or a batch "best".

---

## 5. Method note

The measurements in section 1 came from a throwaway script that plays the game loop from
`training/policy_gradient.py`'s eval worker with extra bookkeeping (asteroid birth frame,
last velocity flip, per-frame straight-line closest-approach for every asteroid, last 400
actions). It is not in the repo. If these diagnostics are wanted as a permanent tool, the
natural home is `tools/analyze_deaths.py` with a README entry.
