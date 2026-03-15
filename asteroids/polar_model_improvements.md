# Proposed Changes to Polar NN State Vector

## Current State (39 inputs)

Global (3): player speed magnitude, shot cooldown, asteroid count
Per asteroid (4 x 9): distance, relative angle from bearing, closing speed, size category

## Proposed Changes

### Remove size category, use edge-to-edge distance instead of center-to-center
- Size category (0.33/0.66/1.0) is hard for the model to learn to use. The heuristic AI doesn't need it.
- Subtract player radius + asteroid radius from distance to get edge-to-edge distance. This directly encodes "how close to collision" and "how close to a hit" — the quantities that actually matter for both dodging and aiming. Big asteroids naturally appear closer (easier to hit, more dangerous) without the model needing to learn what size means.
- Net effect: drop from 4 to 3 per-asteroid features.

### Replace closing speed with time-to-impact
- **Time-to-impact** = distance / closing_speed. More directly useful — answers "how urgent is this?" instead of "how fast is it approaching?" Receding asteroids (negative closing speed) clamped to a large value. Distance is still a separate feature, so closing speed is recoverable if the model needs it.

### Add lateral velocity per asteroid (for aiming)
- The component of asteroid velocity perpendicular to the player-asteroid line. Closing speed only captures the radial component. Without lateral velocity, the model can't predict where the asteroid will be when the bullet arrives. This is the single biggest gap for aiming.

### Add player velocity direction (for dodging)
- The model knows speed magnitude but not drift direction. After accelerating, the player may be drifting sideways into another asteroid with no way to know. Encoded as the angle between velocity vector and facing direction (normalized to [-1, 1]).

### Priority
1. Remove size category + use edge-to-edge distance (simplifies model, better feature)
2. Lateral velocity (biggest impact — helps both aiming and dodging)
3. Replace closing speed with time-to-impact (more useful)
4. Player velocity direction (cheap to add, helps dodging)

### Impact on architecture
- Remove size + add lateral velocity + player velocity direction: (3 per asteroid - 1 + 1) * 9 + 3 global + 1 = 31 inputs
- Replace closing speed with time-to-impact: no change in count
- All changes combined: 31 inputs (down from 39)
- Current 128/64 hidden layers are more than sufficient for 31 inputs
- Requires retraining from scratch (new input dimensions = incompatible weights)
