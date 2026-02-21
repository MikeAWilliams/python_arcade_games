# Analysis: Cross-Entropy Training — Why Loss Improves But Gameplay Doesn't

## Context

The cross-entropy training pipeline trains a neural network to imitate the Smart AI by
predicting its actions from game states. Training loss drops from 1.76 to 1.05 over 400
iterations, but in-game evaluation scores actually *worsen* (from ~4-5 down to ~2) and
the model appears to behave randomly (spinning in place).

---

## 1. The Loss Function Explained

**File:** `training/cross_entropy.py:38-46`

```python
log_probs = F.log_softmax(logits, dim=1)           # normalize logits to log-probabilities
loss = -torch.mean(torch.sum(y * log_probs, dim=1)) # cross-entropy with one-hot targets
```

This is standard categorical cross-entropy. For each sample it computes
`-log(P(correct_action))` — the negative log of the probability the model assigns to
the action the expert actually took. The mean over the batch is the loss.

**Theoretical bounds for a 6-class problem:**

| Scenario | Loss |
|---|---|
| Perfect prediction (always 100% on correct class) | 0.0 |
| Uniform random guess (1/6 each) | **ln(6) = 1.7918** |
| Predicting the marginal class distribution regardless of state | **0.9531** (entropy of the training data action distribution) |

**Your results:**
- Starting loss: **1.764** — virtually identical to random (1.792), expected for an untrained network
- Final loss: **1.047** — only 0.09 above the marginal entropy of 0.953

**Interpretation:** The model has converged almost entirely to predicting the *average
action distribution* irrespective of the game state. It has barely learned any
state-conditional behavior. The loss can't drop much further without the model actually
understanding game context.

---

## 2. Root Cause: Catastrophic Class Imbalance in Training Data

The training data action distribution (from file 0, 10M samples):

| Action | Count | Percentage |
|---|---|---|
| TURN_LEFT | 4,666,960 | **46.6%** |
| TURN_RIGHT | 4,668,487 | **46.7%** |
| SHOOT | 410,584 | 4.1% |
| DECELERATE | 184,583 | 1.8% |
| ACCELERATE | 75,931 | 0.8% |
| NO_ACTION | 0 | 0.0% |

**93.3% of all training labels are TURN_LEFT or TURN_RIGHT.**

This makes sense given the Smart AI's behavior — it spends most of its time in
`SHOOT_NEAREST` strategy, continuously adjusting heading toward the nearest asteroid,
which produces a constant stream of left/right turns. The critical actions
(ACCELERATE, SHOOT) that actually determine game performance are drowned out.

**The model's optimal strategy under cross-entropy with this data is to always output
~47%/47% for TURN_LEFT/TURN_RIGHT with small residuals for other actions.** This
minimizes loss but produces a ship that just spins randomly in place — exactly what
you observe.

---

## 3. Secondary Issues

### 3a. Enormous Batch Size
- Each file has **10M samples**, divided into 3 batches = **3.3M samples per batch**
- At this scale, each gradient step is essentially full-dataset gradient descent
- The gradient is dominated by the majority classes (turning), washing out any
  learning signal from rare but critical actions (shoot, accelerate)

### 3b. Insufficient Training Iterations
- 400 iterations with 345 batches/epoch = only **~1.16 epochs**
- The model barely sees the data once
- But this matters less because the data imbalance is the binding constraint —
  more epochs of imbalanced data would just reinforce the marginal distribution

### 3c. Model Capacity (Moderate Concern)
- Architecture: Linear(141→128, no bias) → ReLU → Linear(128→6, no bias)
- 18,176 parameters total
- A single hidden layer with no bias is a legitimate limitation, but it is
  **not the primary problem** — even a larger model would converge to the same
  marginal distribution with this training data
- The lack of bias terms restricts the model's ability to learn offsets/thresholds

### 3d. Stochastic Sampling at Inference
- `neural.py:120-122` samples from the softmax distribution via `np.random.choice`
- If the model outputs 47%/47%/6%, it will randomly pick turn left or right each
  frame, producing erratic spinning
- Using `argmax` instead of sampling wouldn't help much — it would just
  consistently pick whichever turning direction has a tiny edge

---

## 4. Why Eval Scores *Decrease* During Training

Early in training (iterations 0-70), the model outputs near-uniform probabilities
(~16.7% per action), so it occasionally accelerates and shoots by chance, scoring
avg 3-5. As training progresses, the model learns the training distribution more
precisely (47%/47% turning), so it stops accidentally taking useful actions like
shooting and accelerating. Scores drop to 1.7-2.5.

**Lower loss = better imitation of the data distribution = worse gameplay**, because
the data distribution itself is pathological for gameplay when stripped of
state-conditioning.

---

## 5. Recommended Fixes (Priority Order)

### Fix 1: Rebalance the Training Data
- **Class-weighted loss**: Weight rare actions (SHOOT, ACCELERATE) much higher
  in the loss function so the model is penalized more for getting them wrong
- **Undersample** majority classes or **oversample** minority classes
- **Action grouping**: The Smart AI's turning is mostly a means to aim — consider
  encoding the *intent* (aim-at-target) rather than raw actions

### Fix 2: Reduce Batch Size Dramatically
- Use batches of 256-2048 samples, not 3.3 million
- Smaller batches provide noisier but more useful gradients
- Will require many more iterations (thousands, not hundreds)

### Fix 3: Add Bias Terms to Linear Layers
- Change `bias=False` to `bias=True` in both layers in `neural.py:79,83`
- This is a simple change that adds 134 parameters but meaningfully
  increases representational capacity

### Fix 4: Increase Model Capacity
- Add a second hidden layer: 141 → 256 → 128 → 6
- Consider using bias terms
- Total parameters would grow from 18K to ~70K — still very small

### Fix 5: Train Longer with Proper Batching
- With batch_size=1024 and 10M+ samples, one epoch would be ~10,000 iterations
- Train for 5-10 epochs minimum (50K-100K iterations)
- Add learning rate scheduling (e.g., cosine decay)

### Fix 6: Consider Argmax Instead of Sampling at Inference
- In `neural.py:120-122`, use `torch.argmax` instead of `np.random.choice`
- This only helps once the model actually learns state-conditional behavior
- With current training, this wouldn't help (would just consistently pick one turn direction)

---

## 6. Key Files

| File | Role |
|---|---|
| `training/cross_entropy.py` | Training loop, loss function, data loading |
| `asteroids/ai/neural.py` | Model architecture (NNAIParameters), state encoding, inference |
| `asteroids/core/game.py:8-16` | Action enum definition |
| `nn_checkpoints/training_data20k_combinded_cross_entropy.log` | Training log |

---

## 7. Summary

| Question | Answer |
|---|---|
| Is the loss function correct? | Yes, the cross-entropy implementation is mathematically correct |
| Is the training wrong? | The mechanics work, but the data is catastrophically imbalanced (93% turning) |
| Is it not trained enough? | Partially — only 1.16 epochs — but more training won't help without fixing the imbalance |
| Are model parameters insufficient? | Minor factor — the single hidden layer with no bias is limiting, but the data imbalance is the dominant issue |
| Why does it look random? | The model learns to output ~47% left / ~47% right every frame regardless of state, so the ship just spins |

**The loss dropping from 1.76 to 1.05 is the model learning the marginal action frequencies, not learning gameplay strategy.** The theoretical minimum for just predicting the marginal distribution is 0.953 — the model is almost there at 1.047.

---

## 8. Epoch Time Estimates by Batch Size

### Dataset and measurements

- **Total samples per epoch**: ~1.15 billion (115 files, ~10M samples each)
- **Measured per-iteration time** at batch size ~10K: **40 ms** (includes amortized
  eval overhead)
- **File load cost**: ~2.5 s per file, 115 files per epoch = ~5 min, amortized
  across batches from each file
- For larger batch sizes, GPU data transfer becomes significant (~1.86 GB at 3.3M)

### Estimated epoch times

| Batch Size | Batches/File | Iters / Epoch | Est. Time / Iter | Epoch Time | Gradient Updates |
|---|---|---|---|---|---|
| **3,300,000** | 3 | 348 | ~500 ms* | **~8 min** | 348 |
| 1,000,000 | 10 | 1,150 | ~200 ms* | **~9 min** | 1,150 |
| 100,000 | 100 | 11,500 | ~50 ms* | **~15 min** | 11,500 |
| 10,000 | 1,000 | 115,000 | **40 ms** | **~82 min** | 115,000 |
| 1,024 | 9,772 | 1,120,000 | ~35 ms* | **~11.4 hrs** | 1,120,000 |
| 256 | 39,088 | 4,490,000 | ~35 ms* | **~44 hrs** | 4,490,000 |

\* Estimated. Only 10K batch size (40 ms) is measured. Larger batches are slower due
to GPU data transfer (~1.86 GB at 3.3M). Smaller batches approach a floor set by
Python/PyTorch per-iteration overhead.

### Key takeaway

**Batch sizes of 100K–1M are the sweet spot.** They provide 3–33× more gradient
updates than the current 3.3M batch at comparable epoch times (9–15 min vs 8 min).
At batch size 10K, you get 330× more updates but epoch time grows to ~82 min — still
practical for serious training runs.

---

## 9. State Vector Data Range Analysis

Sampled 500K rows (evenly spaced) from file 0 (10M samples total).
State vector has 141 columns: 6 player features + 27 asteroids × 5 features each.

All values are normalized by game dimensions (1280×720) or relevant constants.
See `asteroids/ai/neural.py:compute_state()` for the encoding.

### Player state (columns 0–5)

| Col | Feature | Min | Max | Mean | Std | Notes |
|---|---|---|---|---|---|---|
| 0 | player_x | 0.016 | 0.984 | 0.498 | 0.143 | Well centered, full range |
| 1 | player_y | 0.028 | 0.972 | 0.500 | 0.160 | Well centered, full range |
| 2 | player_vx | -0.163 | 0.163 | 0.000 | 0.028 | Narrow range, centered |
| 3 | player_vy | -0.295 | 0.292 | 0.000 | 0.052 | Wider than vx (~1.8×) |
| 4 | **player_angle** | **-4.350** | **3.717** | **0.253** | **0.633** | **PROBLEM: unbounded** |
| 5 | shoot_cooldown | 0.000 | 0.950 | 0.390 | 0.318 | [0, 1] range as expected |

### Asteroid state (columns 6–140, grouped by type)

Since all 27 asteroid slots follow the same pattern, stats are summarized by
asteroid generation. Inactive asteroids contribute all-zero rows, pulling means low.

| Group | Slots | Active % | x Mean | y Mean | vx Std | vy Std |
|---|---|---|---|---|---|---|
| Big (0–2) | 3 | ~21% | 0.107 | 0.105 | 0.032 | 0.063 |
| Medium (3–8) | 6 | ~15% | 0.076 | 0.077 | 0.029 | 0.051 |
| Small (9–26) | 18 | ~19% | 0.095 | 0.095 | 0.032 | 0.056 |

Position means are low (~0.08–0.11) because inactive slots are all zeros.
When filtering to active-only, positions would center around ~0.5.

Velocity ranges for asteroids:
- vx: [-0.184, 0.184] (all generations similar)
- vy: [-0.350, 0.350] (big asteroids slightly wider)

### Key finding: player_angle is not normalized to [0, 1]

The angle is computed as `game.player.geometry.angle / (2 * math.pi)`, which should
yield values in [0, 1]. However, the data shows a range of **[-4.35, 3.72]**, meaning
the raw angle spans approximately [-27.3, 23.4] radians.

**The player angle accumulates without wrapping.** The game does not normalize the
angle to [0, 2π] — it just adds/subtracts the turn rate each frame, so the angle grows
or shrinks unboundedly over the lifetime of a game.

This is a problem for the neural network because:
1. **Same physical angle, different values**: An angle of 0 and 2π (normalized: 0 and 1)
   point in the same direction but the model sees them as completely different inputs
2. **Unbounded range**: Most features are in [0, 1] but angle spans [-4.35, 3.72],
   giving it outsized influence on the first linear layer
3. **Non-stationary**: The angle value depends on how long the game has been running
   and how much the player has turned, not just the current heading

### Scalar angle vs (sin, cos) heading encoding

The two options for fixing the angle are:
1. **Scalar [0, 1]**: wrap angle to [0, 2π], then divide by 2π
2. **(sin θ, cos θ)**: encode heading as two values, each in [-1, 1]

**(sin θ, cos θ) is significantly better for this model.** Here's why:

The core task the angle serves is deciding "turn left, turn right, or shoot?" — which
depends on the angular relationship between the player's heading and the direction to
asteroids. The Smart AI's dominant SHOOT_NEAREST strategy is: compute angle to nearest
asteroid, compare to heading, turn toward it, shoot when aligned.

**Scalar angle in [0, 1]:**
- Has a discontinuity at the wrap point: 0.001 and 0.999 are almost the same heading
  but numerically distant. A linear layer treats them as maximally different.
- The first linear layer computes `w * angle + ...` — a linear function of a scalar
  angle cannot represent "am I facing left of the asteroid?" because that relationship
  is sinusoidal, not linear.
- The hidden layer must learn sin/cos-like patterns from a single scalar through ReLU.
  This is possible in theory but wastes hidden layer capacity (neurons spent
  approximating trig functions rather than learning gameplay strategy).

**(sin θ, cos θ) encoding:**
- No discontinuity — nearby angles always produce nearby values. Wrapping is natural.
- A single linear neuron can compute `w1 * cos θ + w2 * sin θ = R * cos(θ - φ)` for
  some learned magnitude and phase. This is exactly the operation needed to compare
  headings.
- To determine if an asteroid at relative position (dx, dy) is to the left or right
  of heading (cos θ, sin θ), you need the cross product:
  `dx * sin θ - dy * cos θ`. **A single linear neuron can learn this directly** from
  the (sin θ, cos θ) inputs combined with asteroid position inputs.
- With a scalar angle, the network must first reconstruct sin and cos from θ using
  ReLU activations before it can reason about aiming.

**For this specific model** (141→128→6, no bias, ReLU): the single hidden layer is the
bottleneck. Spending neurons learning to approximate trig functions from a scalar
leaves fewer neurons for actual strategy. With (sin, cos), the first linear layer can
directly compute angular relationships in a single matrix multiply.

The tradeoff is one extra input dimension (142 vs 141), which is negligible.

---

## 10. Heuristic AI Tick Cycle Estimates

Understanding how many game ticks the heuristic AI spends in each behavioral cycle
helps inform batch size choices — a batch should contain enough samples to capture
complete decision cycles, not just fragments of turning.

### Fundamental rates

- **Turn rate**: `PLAYER_TURN_RATE` = π rad/s → π/60 ≈ 0.0524 rad/tick ≈ 3°/tick
- **Acceleration**: 180 px/s² → 3 px/s per tick
- **Game tick**: dt = 1/60 s

### SHOOT_NEAREST cycle (turn → fire)

The AI turns toward the predicted intercept and fires when `|angle_diff| < shoot_angle_tolerance` (0.425 rad ≈ 24°).

- Tolerance window: 0.425/π ≈ 13.5% chance already aimed
- Average angle to turn (when not aimed): (π − 0.425)/2 ≈ 1.36 rad → ~26 ticks
- Weighted average including already-aimed: ~22 ticks turning
- Plus 20-tick `SHOOT_COOLDOWN` before next shot

**Typical shoot cycle: ~23 ticks turning + fire + 20 tick cooldown ≈ 43 ticks (~0.7s)**

During cooldown the AI keeps turning toward the next target, so consecutive shots overlap.

### EVASIVE_ACTION cycle (detect threat → thrust away)

Two phases: turn to align, then thrust (accelerate or decelerate).

**Phase 1 — Turning**: Needs `|angle_diff|` either < 0.682 rad (to decel) or > π − 0.682 (to accel), whichever is closer.
- Combined tolerance zones cover 2 × 0.682 = 1.364 rad out of π → 43% chance already aligned
- Worst-case turn (from π/2): 0.889 rad → 17 ticks
- Average turn when needed: ~0.445 rad → ~9 ticks
- Weighted average: ~5 ticks

**Phase 2 — Thrusting**: Continues while asteroids are predicted to collide within the 49-tick lookahead window. Each tick of thrust changes velocity by 3 px/s, so multiple ticks are needed to deflect trajectory enough.

**Typical evasion cycle: 5–17 ticks turning + 10–30 ticks thrusting ≈ 15–45 ticks (~0.25–0.75s)**

Upper bound naturally capped by the 49-tick lookahead.

### SPEED_CONTROL cycle

Same turning mechanics as evasion (0.682 rad tolerance), then decelerate until speed drops below `max_speed` (765 px/s).

**Typical cycle: 5–9 ticks turning + 10–20 ticks decelerating ≈ 15–29 ticks**

### Summary

| Cycle | Turn Ticks | Action Ticks | Total | Real Time |
|---|---|---|---|---|
| Shoot (average) | ~22 | 1 + 20 cooldown | ~43 | ~0.7s |
| Evasion (typical) | 5–17 | 10–30 | 15–45 | 0.25–0.75s |
| Speed control | 5–9 | 10–20 | 15–29 | 0.25–0.5s |

### Batch size consideration

A batch size should be large enough to contain multiple complete decision cycles,
not just fragments of turning. At ~43 ticks per shoot cycle and ~30 ticks per evasion
cycle, a single game tick carries very little independent signal — it's almost always
"continue turning in the same direction." This reinforces the class imbalance problem
(Section 2): the 93% turning labels aren't 93% independent decisions, they're long
runs of the same turn action repeated 20+ times in a row.

A batch of 256 samples drawn from a single game likely contains only 6–8 complete
decision cycles. A batch of 1024 contains ~24–30 cycles. Batches of 10K+ contain
hundreds of cycles across multiple games, giving a more representative gradient.

However, extremely large batches (3.3M) average out the signal from rare actions
(Section 3a). The sweet spot from Section 8 (100K–1M) provides enough complete cycles
for stable gradients while preserving learning signal from minority actions.
