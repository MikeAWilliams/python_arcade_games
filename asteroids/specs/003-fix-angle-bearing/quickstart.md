# Quickstart: Fix Angle Normalization and Bearing Encoding

## What Changed

1. **Player angle** is now normalized to [0, 2pi) in game.py
2. **NN state vector** encodes heading as (cos, sin) instead of scalar angle (141 -> 142 inputs)
3. **Model loading** validates parameter dimensions and errors on mismatch
4. **New tool** `tools/convert_training_data.py` converts old training data to new format

## After Implementation

### Convert existing training data

```bash
python tools/convert_training_data.py --input-base training_data20k_combinded --output-base training_data20k_v2
```

### Verify conversion with analysis tool

```bash
python tools/analyze_state_data.py --base-name training_data20k_v2
```

Columns 4 and 5 should show values in [-1, 1] (cos and sin of angle).

### Retrain a model

```bash
python training/cross_entropy.py --base-name training_data20k_v2
```

### Run with new model

```bash
python main_arcade.py --ain <new_model_file.pth>
```

### Old models will not load

Attempting to load a model trained on 141-input data will produce an error like:
```
Model incompatible: first layer expects 142 inputs but loaded weights have 141.
```

## Files Modified

| File | Change |
| ---- | ------ |
| `asteroids/core/game.py` | Angle normalization in `Player.update()` |
| `asteroids/ai/neural.py` | Bearing encoding, input dim 142, validation helper |
| `main_arcade.py` | Use validation helper for model loading |
| `main_headless.py` | Use validation helper for model loading |
| `training/cross_entropy.py` | Use validation helper for model loading |
| `training/policy_gradient.py` | Use validation helper for model loading |
| `tools/analyze_state_data.py` | Updated column names for bearing |
| `tools/convert_training_data.py` | New file - data conversion tool |
