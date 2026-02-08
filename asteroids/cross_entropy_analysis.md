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

### Baseline measurements

- **Total samples per epoch**: ~1.15 billion (115 files, ~10M samples each)
- **Current batch size**: ~3.3M (10M per file / 3 batches per file)
- **Current iterations per epoch**: ~345 (115 files × 3 batches)
- **Observed time**: ~1 hour for 400 iterations (~9s/iter), which is just over 1 epoch
- **Estimated epoch time at current batch size**: ~300 iterations × 9s ≈ **45 minutes**

### Empirical finding: per-iteration time is constant

A test run with `--batch-per-file 1000` (batch size ~10K, a 330× reduction) shows
nearly identical per-iteration time:

| Run | Batch/File | Batch Size | Time/Iter |
|---|---|---|---|
| Original | 3 | ~3,300,000 | ~9.1 s |
| Test | 1,000 | ~10,000 | ~8.0 s |

**Per-iteration time does NOT scale with batch size.** It is dominated by a constant
overhead: the `NpzFile` decompression in `get_batch()`.

### Root cause: NPZ decompression on every array access

`np.load()` returns a lazy `NpzFile` object. Every access to `self.data["states"]`
decompresses the full compressed array from disk (~60 MB compressed → ~5.6 GB
uncompressed). In `get_batch()` this happens multiple times per call:

```python
# Line 89-90: boundary check — decompresses full "states" array just for len()
if self.batch_index * self.batch_size + self.batch_size >= len(self.data["states"]):

# Line 99-100: actual slice — decompresses both arrays again
states = self.data["states"][start:end]
labels = self.data["actions"][start:end]
```

This ~8-second decompression cost is paid every iteration regardless of batch size.
The actual GPU compute for a batch through the 18K-parameter model is negligible
(< 1 ms even at 3.3M batch size).

### Epoch times before the fix (broken data loader)

With constant ~8 s/iter, epoch time scaled linearly with iteration count.
Smaller batches were completely impractical.

| Batch Size | Batches/File | Iters / Epoch | Time / Iter | Epoch Time | Gradient Updates |
|---|---|---|---|---|---|
| **3,300,000** | 3 | 348 | ~9 s | **~52 min** | 348 |
| 1,000,000 | 10 | 1,150 | ~8 s | **~2.6 hrs** | 1,150 |
| 100,000 | 100 | 11,500 | ~8 s | **~26 hrs** | 11,500 |
| 10,000 | 1,000 | 115,000 | ~8 s | **~11 days** | 115,000 |
| 1,024 | 9,772 | 1,120,000 | ~8 s | **~104 days** | 1,120,000 |
| 256 | 39,088 | 4,490,000 | ~8 s | **~1.1 years** | 4,490,000 |

### The fix: cache decompressed arrays

The fix was to decompress each npz file once in `load_data()` and cache the resulting
numpy arrays, instead of re-decompressing on every `get_batch()` access:

```python
def load_data(self):
    file = self.files[self.file_index]
    raw = np.load(file)
    self.states = raw["states"]    # decompress once, cache as numpy array
    self.actions = raw["actions"]  # decompress once, cache as numpy array
    self.batch_size = len(self.states) // self.batch_per_file
    self.batch_index = 0
```

### Measured result: 200× speedup

With the fixed data loader at `--batch-per-file 1000` (batch size ~10K):

| Run | Batch/File | Batch Size | Time/Iter | 400 Iters |
|---|---|---|---|---|
| Before fix | 1,000 | ~10,000 | ~8.0 s | ~53 min |
| **After fix** | 1,000 | ~10,000 | **~40 ms** | **16.1 s** |

Per-iteration time dropped from ~8 seconds to **~40 ms** — a **200× speedup**.
The 400 iterations completed in 16.1 seconds total (was ~53 minutes).

### Epoch times after the fix

Measured per-iteration time at batch_size ~10K: **40 ms** (includes amortized eval
overhead). File load cost is ~2.5 s per file, 115 files per epoch = ~5 min, amortized
across batches from each file. For larger batch sizes, GPU data transfer becomes
significant (~1.86 GB at 3.3M batch).

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

**Batch sizes of 100K–1M are now the sweet spot.** They provide 3–33× more gradient
updates than the current 3.3M batch at comparable epoch times (9–15 min vs 8 min).
At batch size 10K, you get 330× more updates but epoch time grows to ~82 min — still
practical and far better than the 11 days it would have taken before the fix.
