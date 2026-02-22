# Data Model: Fix Angle Normalization and Bearing Encoding

## Entities

### State Vector (Neural Network Input)

The numerical representation of game state fed to the neural network.

**Old layout** (141 dimensions):

| Index | Field | Range | Description |
| ----- | ----- | ----- | ----------- |
| 0 | player_x | [0, 1] | Normalized x position |
| 1 | player_y | [0, 1] | Normalized y position |
| 2 | player_vx | unbounded | Normalized x velocity |
| 3 | player_vy | unbounded | Normalized y velocity |
| 4 | player_angle | unbounded | angle / (2*pi), unbounded |
| 5 | shoot_cooldown | [0, 1] | Normalized cooldown |
| 6-140 | asteroid data | mixed | 27 asteroids x 5 features |

**New layout** (142 dimensions):

| Index | Field | Range | Description |
| ----- | ----- | ----- | ----------- |
| 0 | player_x | [0, 1] | Normalized x position |
| 1 | player_y | [0, 1] | Normalized y position |
| 2 | player_vx | unbounded | Normalized x velocity |
| 3 | player_vy | unbounded | Normalized y velocity |
| 4 | bearing_x | [-1, 1] | cos(angle) |
| 5 | bearing_y | [-1, 1] | sin(angle) |
| 6 | shoot_cooldown | [0, 1] | Normalized cooldown |
| 7-141 | asteroid data | mixed | 27 asteroids x 5 features |

### Player Angle

| Property | Old | New |
| -------- | --- | --- |
| Storage | Unbounded float (radians) | Float in [0, 2*pi) |
| Normalization | None | `angle % (2 * math.pi)` after each update |
| NN encoding | `angle / (2*pi)` (1 value) | `cos(angle), sin(angle)` (2 values) |

### Training Data File (.npz)

| Array Key | Type | Old Shape | New Shape | Changed? |
| --------- | ---- | --------- | --------- | -------- |
| states | float32 | (N, 141) | (N, 142) | Yes |
| actions | int8 | (N,) | (N,) | No |
| game_ids | int32 | (N,) | (N,) | No |
| tick_nums | int32 | (N,) | (N,) | No |

### Model Architecture

| Layer | Old Shape | New Shape |
| ----- | --------- | --------- |
| Linear 1 (input) | 141 x 128 | 142 x 128 |
| ReLU | - | - |
| Linear 2 (output) | 128 x 6 | 128 x 6 |
