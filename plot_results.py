import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results.csv")))
conditions = list(dict.fromkeys(r["condition"] for r in rows))
controllers = ["PID", "PPO (nominal)", "PPO (DR)"]   # open-loop left out: it doesn't track
colors = {"PID": "tab:orange", "PPO (nominal)": "tab:blue", "PPO (DR)": "tab:green"}  # same as plot_tracking.py
get = lambda c, k, field: next(float(r[field]) for r in rows
                               if r["condition"] == c and r["controller"] == k)

x = np.arange(len(conditions)); w = 0.27
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for j, k in enumerate(controllers):
    ax[0].bar(x + (j-1)*w, [get(c, k, "rms_mm") for c in conditions], w, label=k, color=colors[k])
    ax[1].bar(x + (j-1)*w, [get(c, k, "chatter_V_per_step") for c in conditions], w, label=k, color=colors[k])
for a in ax:
    a.axvspan(2.5, len(conditions) - 0.5, color="gray", alpha=0.12)   # out-of-range region
ax[0].text(5.0, ax[0].get_ylim()[1]*0.92, "Outside training range", ha="center")
ax[0].set_ylabel("RMS tracking error [mm]"); ax[1].set_ylabel("Voltage chatter [V/step]")
ax[1].set_xticks(x); ax[1].set_xticklabels(conditions, rotation=25, ha="right")
ax[0].legend(); ax[0].set_title("Controller comparison across actuator conditions (10 episodes each)")
plt.tight_layout(); plt.savefig("figures/results_comparison.png", dpi=150); plt.show()