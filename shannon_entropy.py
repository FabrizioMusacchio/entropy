# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
# %% EXAMPLE: TEXT MESSAGE
def calculate_entropy(pmf):
    entropy = 0
    for p in pmf:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

# Sample dataset
data = ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B']

# Convert categories to numerical values
categories = list(set(data))
numerical_data = [categories.index(d) for d in data]

# Calculate PMF
pmf = np.bincount(numerical_data) / len(numerical_data)

# Calculate Shannon entropy
shannon_entropy = calculate_entropy(pmf)

# Display results
print("Shannon Entropy:", shannon_entropy)

# Plotting the PMF
plt.figure(figsize=(4, 4))
plt.bar(range(len(pmf)), pmf)
plt.xlabel("Categories")
plt.ylabel("Probability")
plt.title("Probability Mass Function (PMF)")
plt.xticks(range(len(pmf)), categories)
plt.tight_layout()
plt.savefig("shannon_categorial_example.png", dpi=200)
plt.show()
# %% EXAMPLE: ACTION POTENTIALS
np.random.seed(0)

# Simulate two spike trains for 1000 ms
T = 1000
N_spikes_1 = 25
N_spikes_2 = 100
spike_train_1 = np.zeros(T)
spike_train_2 = np.zeros(T)
spike_times_1 = np.random.choice(T, size=N_spikes_1, replace=False)
spike_times_2 = np.random.choice(T, size=N_spikes_2, replace=False)
spike_train_1[spike_times_1] = 1
spike_train_2[spike_times_2] = 1

# Calculate entropy and information rate
counts_1 = np.bincount(spike_train_1.astype(int))
probs_1 = counts_1 / float(sum(counts_1))
H_1 = entropy(probs_1, base=2)
info_rate_1 = H_1 / (T / 1000)

counts_2 = np.bincount(spike_train_2.astype(int))
probs_2 = counts_2 / float(sum(counts_2))
H_2 = entropy(probs_2, base=2)
info_rate_2 = H_2 / (T / 1000)

# Plot spike trains and information rates
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].eventplot(spike_times_1, color='C0')
axs[0].set_title(f"Spike Train 1: N_spikes={N_spikes_1}, information rate={info_rate_1:.2f} bits/sec")
axs[0].set_ylabel("Spikes")
axs[0].set_xlabel("Time (ms)")
axs[1].eventplot(spike_times_2, color='C1')
axs[1].set_title(f"Spike Train 2: N_spikes={N_spikes_2}, information rate={info_rate_2:.2f} bits/sec")
axs[1].set_ylabel("Spikes")
axs[1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig("shannon_spike_trains.png", dpi=200)
plt.show()
# %%END