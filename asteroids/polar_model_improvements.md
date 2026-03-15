# Proposed Changes to Polar NN State Vector

## Current State (39 inputs)

Global (3): player speed magnitude, shot cooldown, asteroid count
Per asteroid (4 x 9): distance, relative angle from bearing, closing speed, size category

## Proposed Changes

### Replace closing speed with time-to-impact (no net change in feature count)
- **Time-to-impact** = distance / closing_speed. More directly useful than closing speed — answers "how urgent is this?" instead of "how fast is it approaching?" Receding asteroids (negative closing speed) clamped to a large value. Distance is still a separate feature, so closing speed is recoverable if the model needs it.

### Add lateral velocity per asteroid (for aiming)
- The component of asteroid velocity perpendicular to the player-asteroid line. Closing speed only captures the radial component. Without lateral velocity, the model can't predict where the asteroid will be when the bullet arrives. This is the single biggest gap for aiming.

### Add player velocity direction (for dodging)
- The model knows speed magnitude but not drift direction. After accelerating, the player may be drifting sideways into another asteroid with no way to know. Encoded as the angle between velocity vector and facing direction (normalized to [-1, 1]).

### Priority
1. Lateral velocity (biggest impact — helps both aiming and dodging)
2. Replace closing speed with time-to-impact (same feature count, more useful)
3. Player velocity direction (cheap to add, helps dodging)

### Impact on architecture
- Replace closing speed + add lateral velocity + player velocity direction: 39 + 9 + 1 = 49 inputs
- May need to increase hidden layer sizes for the larger input, but the current 128/64 should handle 49 inputs fine
- Requires retraining from scratch (new input dimensions = incompatible weights)
