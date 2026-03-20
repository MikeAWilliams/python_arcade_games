"""
Visualize the effect of death penalty reward shaping on discounted returns.

Simulates a game where 27 asteroids are killed spread over ~5000 frames,
then the player dies. Shows raw rewards, the death penalty ramp, and
discounted returns with and without the penalty.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discounted_returns(rewards, gamma=0.99, normalize=True):
    eps = 0.0001
    ret = np.zeros(len(rewards))
    s = 0
    for i in range(len(rewards) - 1, -1, -1):
        s = rewards[i] + gamma * s
        ret[i] = s
    if normalize:
        ret = (ret - np.mean(ret)) / (np.std(ret) + eps)
    return ret


def main():
    # 27 asteroids killed spread over ~5000 frames, then death
    # Score is +1 per asteroid regardless of size
    np.random.seed(42)
    n_frames = 5000
    rewards = np.zeros(n_frames)

    kill_frames = np.sort(
        np.random.choice(range(100, n_frames - 100), 27, replace=False)
    )
    for f in kill_frames:
        rewards[f] = 1

    survival_bonus = 0.001
    rewards_no_penalty = rewards + survival_bonus

    # Death penalty version
    death_penalty_frames = 60
    death_penalty = -0.1
    rewards_with_penalty = rewards_no_penalty.copy()
    for i in range(
        max(0, len(rewards_with_penalty) - death_penalty_frames),
        len(rewards_with_penalty),
    ):
        decay = (
            i - (len(rewards_with_penalty) - death_penalty_frames)
        ) / death_penalty_frames
        rewards_with_penalty[i] += death_penalty * decay

    dr_no = discounted_returns(rewards_no_penalty)
    dr_yes = discounted_returns(rewards_with_penalty)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    # 1: Raw kill rewards
    axes[0].stem(
        np.where(rewards > 0)[0],
        rewards[rewards > 0],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
    )
    axes[0].set_ylabel("Points")
    axes[0].set_title("Raw kill rewards (27 asteroids over 5000 frames)")
    axes[0].set_xlim(0, n_frames)

    # 2: Last 200 frames raw rewards zoomed
    x = np.arange(n_frames)
    axes[1].plot(x, rewards_no_penalty, "C0-", alpha=0.8, label="No death penalty")
    axes[1].plot(x, rewards_with_penalty, "C3-", alpha=0.8, label="With death penalty")
    axes[1].set_xlim(n_frames - 200, n_frames)
    axes[1].set_ylim(-0.6, 0.05)
    axes[1].axvline(n_frames - 60, color="gray", linestyle="--", alpha=0.6)
    axes[1].text(
        n_frames - 60 - 2,
        -0.25,
        "penalty\nstarts",
        ha="right",
        fontsize=9,
        color="gray",
    )
    axes[1].set_ylabel("Raw reward")
    axes[1].set_title("Last 200 frames: raw reward with death penalty ramp")
    axes[1].legend()

    # 3: Discounted returns WITHOUT penalty
    axes[2].plot(dr_no, "C0-", linewidth=0.7)
    axes[2].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("Advantage")
    axes[2].set_title("Discounted returns WITHOUT death penalty (normalized)")
    axes[2].set_xlim(0, n_frames)

    # 4: Discounted returns WITH penalty
    axes[3].plot(dr_yes, "C3-", linewidth=0.7)
    axes[3].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[3].set_ylabel("Advantage")
    axes[3].set_title("Discounted returns WITH death penalty (normalized)")
    axes[3].set_xlabel("Frame")
    axes[3].set_xlim(0, n_frames)

    plt.tight_layout()
    output_path = "reward_shaping_comparison.png"
    plt.savefig(output_path, dpi=120)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
