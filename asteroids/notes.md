average over 5000 games with default parameters and simple evasion strategy is 88.1, 307 games per second
after new look ahead strategy, 60 ticks look ahead average score is 101, 18 games per second
use dist2 improves to 26 games per second
optimize the loop by using local variables instead of members gets to 60.5 games per second
current default parameters
        evasion_max_distance: float = 550,
        max_speed: float = 100,
        evasion_lookahead_ticks: int = 60,
        shoot_angle_tolerance: float = 0.05,
        movement_angle_tolerance: float = 0.1,

movement_angle_tolerance
  1 - average 103
  0.01 - average 52

shoot_angle_tolerance
  1 - average 102
  0.01 - average 67

evasion_lookahead_ticks
  120 - average 81
  30 - average 93

max_speed
  200 - average 119
  50 - average 65

evasion_max_distance
  1100 - average 100
  200 - average 100
  50 - average 100
  10 - average 98
---------------------------------------------------------------------------------------
main_genetic with 10 games per individual, 50 individuals and 20 generations produced
Total evolution time: 198.1 seconds
Best fitness: 177.2
Best parameters:
  evasion_max_distance:     436
  max_speed:                490
  evasion_lookahead_ticks:  53
  shoot_angle_tolerance:    0.407
  movement_angle_tolerance: 0.680

5000 games with the above stats yields an average score of 134.7
---------------------------------------------------------------------------------------
Population: 50, Generations: 20, Games per individual: 50
Total evolution time: 925.2 seconds
Best fitness: 147.1

Best parameters:
  evasion_max_distance:     637
  max_speed:                518
  evasion_lookahead_ticks:  50
  shoot_angle_tolerance:    0.503
  movement_angle_tolerance: 0.549

5000 games with the above stats yields an average score of 133.3

---------------------------------------------------------------------------------------

Population: 250, GTotal evolution time: 3481.4 seconds
Best fitness: 157.4

Best parameters:
  evasion_max_distance:     579
  max_speed:                765
  evasion_lookahead_ticks:  49
  shoot_angle_tolerance:    0.425
  movement_angle_tolerance: 0.682

5000 games with the above stats yields an average score of 136.2

---------------------------------------------------------------------------------------
nn training record 25-jan 8:30 pm after most recent commit
python3 main_pg.py --batch-size 320 --workers 16
Using device: cpu
Using 16 worker processes for game simulation
Batch size: 320 games per training update
Creating persistent worker pool...
0/2000 -> avg_score:4.17, max:4.17 | sim:1.18s, train:0.10s | elapsed:00:00:01, total:00:42:41, remaining:00:42:39
100/2000 -> avg_score:4.38, max:5.30 | sim:1.26s, train:0.12s | elapsed:00:02:17, total:00:45:27, remaining:00:43:09
200/2000 -> avg_score:4.38, max:5.30 | sim:1.05s, train:0.10s | elapsed:00:04:33, total:00:45:19, remaining:00:40:46
300/2000 -> avg_score:4.52, max:5.31 | sim:1.29s, train:0.12s | elapsed:00:06:51, total:00:45:31, remaining:00:38:40
1400/2000 -> avg_score:4.17, max:5.31 | sim:1.24s, train:0.11s | elapsed:00:09:07, total:00:45:32, remaining:00:36:24
500/2000 -> avg_score:3.98, max:5.33 | sim:1.18s, train:0.10s | elapsed:00:11:26, total:00:45:41, remaining:00:34:15
600/2000 -> avg_score:4.25, max:5.78 | sim:1.10s, train:0.09s | elapsed:00:13:47, total:00:45:52, remaining:00:32:05
700/2000 -> avg_score:3.96, max:5.78 | sim:1.22s, train:0.10s | elapsed:00:16:10, total:00:46:10, remaining:00:29:59
800/2000 -> avg_score:4.30, max:5.78 | sim:1.30s, train:0.11s | elapsed:00:18:32, total:00:46:16, remaining:00:27:44
900/2000 -> avg_score:4.09, max:5.78 | sim:1.13s, train:0.11s | elapsed:00:20:47, total:00:46:09, remaining:00:25:21
1000/2000 -> avg_score:3.91, max:5.78 | sim:1.08s, train:0.10s | elapsed:00:23:04, total:00:46:06, remaining:00:23:01
1100/2000 -> avg_score:4.50, max:5.78 | sim:1.14s, train:0.10s | elapsed:00:25:24, total:00:46:10, remaining:00:20:45
1200/2000 -> avg_score:4.96, max:5.78 | sim:1.51s, train:0.14s | elapsed:00:27:42, total:00:46:08, remaining:00:18:25
1300/2000 -> avg_score:4.53, max:5.78 | sim:1.23s, train:0.11s | elapsed:00:30:02, total:00:46:11, remaining:00:16:08
1400/2000 -> avg_score:4.60, max:5.78 | sim:1.23s, train:0.11s | elapsed:00:32:21, total:00:46:12, remaining:00:13:50
1500/2000 -> avg_score:4.46, max:5.78 | sim:1.32s, train:0.11s | elapsed:00:34:37, total:00:46:07, remaining:00:11:30
1600/2000 -> avg_score:4.07, max:5.78 | sim:1.21s, train:0.10s | elapsed:00:36:51, total:00:46:03, remaining:00:09:11
1700/2000 -> avg_score:4.14, max:5.78 | sim:1.14s, train:0.10s | elapsed:00:39:08, total:00:46:00, remaining:00:06:52
1800/2000 -> avg_score:4.39, max:5.78 | sim:1.21s, train:0.11s | elapsed:00:41:22, total:00:45:57, remaining:00:04:34
1900/2000 -> avg_score:4.76, max:5.78 | sim:1.27s, train:0.12s | elapsed:00:43:38, total:00:45:54, remaining:00:02:16
Training completed in 00:45:54

stopped improving after 600 iterations
-------------------------------------------------------------------------------------
asked claude to just fix the algorithm. see what we get 9:10 pm
output ran for 7 hours 60k epochs. showed very gradual improvement in the max. Ran in terminal only so output was lost
saved results to nn_best_60k.pth
running in gui mode after the fact and loading the nn_best_60k.pth file revils that it has learned to spin in a cirlce and shoot randomly. This is a major win. That isnt't enough to win but shows it learned
running 1000 runs with it in headless shows an average of about 10 and a max of 109

------------------------------------------------------------------------------------------------
## after changing angle to bearing policy graident results

10% of a policy graident run results in turning and shooting. It sometimes accelerates or decelerates.
Total Games: 1000

SCORE:
  Min:            0
  Max:           66
  Average:      9.2
  Median:       6.0
  Std Dev:      9.1

SURVIVAL TIME (seconds):
  Min:         0.87
  Max:       132.48
  Average:    13.35
  Median:      8.15
  
  checkpoints at 20 and 30 percent do the same
abort the policy gradient run after 30 percent

---------------------------------------------------------------------------------------------
## Rand cross_entrpy for 5 epocs

Saved the logs but it learned the same as above. It rotates and randomly shoots. Learned that in the first 500 iterations and didn't improve much after that

## Try weighting the actions proportionally to their occrence.
logic is
The normalization ensures each class contributes equally to the total gradient. You can verify: frequency × weight ≈ constant for every class:

  ┌────────────┬───────────┬────────┬─────────┐
  │   Action   │ Frequency │ Weight │ Product │
  ├────────────┼───────────┼────────┼─────────┤
  │ TURN_LEFT  │ 46.6%     │ 0.0499 │ ~0.023  │
  ├────────────┼───────────┼────────┼─────────┤
  │ TURN_RIGHT │ 46.7%     │ 0.0499 │ ~0.023  │
  ├────────────┼───────────┼────────┼─────────┤
  │ ACCELERATE │ 0.77%     │ 3.0697 │ ~0.024  │
  ├────────────┼───────────┼────────┼─────────┤
  │ DECELERATE │ 1.87%     │ 1.2628 │ ~0.024  │
  ├────────────┼───────────┼────────┼─────────┤
  │ SHOOT      │ 4.10%     │ 0.5677 │ ~0.023  │
  └────────────┴───────────┴────────┴─────────┘

  So each of the 5 present classes contributes roughly 20% of the total gradient. That's exactly the goal — the model will spend equal "effort" learning when to shoot as when to turn, even though shooting is 11x rarer.

  One thing to be aware of: ACCELERATE at 3.07 is aggressive. It's only 0.77% of the data, so those are relatively few training examples getting 61x amplification compared to turns. If the heuristic AI's acceleration
  decisions are at all noisy or situation-dependent, that signal could be amplified too. Worth watching in the early loss curve — if it's unstable, softening ACCELERATE specifically (e.g. capping weights at some max)
  might help.
