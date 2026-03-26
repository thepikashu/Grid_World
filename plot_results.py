# visualise DQN training progress

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "02_Deep_reinforcement_learning/results/training_log.csv",
    names=["episode", "reward", "epsilon"]
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Reward curve
ax1.plot(df["episode"], df["reward"], color="steelblue", linewidth=1.5)
ax1.set_ylabel("Total Reward")
ax1.set_title("Yashasvee — DQN GridWorld Training")
ax1.grid(True, alpha=0.3)

# Epsilon decay curve
ax2.plot(df["episode"], df["epsilon"], color="coral", linewidth=1.5)
ax2.set_ylabel("Epsilon")
ax2.set_xlabel("Episode")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print("Saved training_curve.png")
