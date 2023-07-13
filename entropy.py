"""
A script to demonstrate the concept of entropy.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: Feb 02, 2023

For reproducibility:

```powershell
conda create -n entropy -y python=3.9
conda activate entropy
conda install -y mamba
mamba install -y ipykernel numpy matplotlib
```
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% VARYING THE NUMBER OF MICROSTATES
def calculate_entropy(microstates):
    entropy = np.log(microstates)
    return entropy

# Sample data of microstates
microstates = np.array([10, 20, 30, 15, 25])

# Calculate entropy using Boltzmann's formula
entropy = calculate_entropy(microstates)

# Display results
print("Entropy:", entropy)

# Plotting the entropy
plt.plot(range(len(microstates)), entropy)
plt.xlabel("System")
plt.ylabel("Entropy")
plt.title("Entropy of the Gas Particle System")
plt.tight_layout()
plt.savefig("entropy.png", dpi=200)
plt.show()
# %% TIME EVOLUTION OF THE ENTROPY

# set random seed for reproducibility
np.random.seed(41)

# Initialize the gas particle system
time_steps = 61  # 60 seconds + 1 (for indexing)
initial_microstates = 5
microstates = [initial_microstates]

# Simulate the evolution of the gas particle system
for t in range(1, time_steps):
    # Update the number of microstates based on a specific process
    if t <= 55:
        new_microstates = microstates[t-1] + np.random.randint(-2, 3)
    else:
        new_microstates = microstates[t-1] - 1  # Entropy decrease 5 seconds before the end
    microstates.append(new_microstates)

# Calculate entropy using Boltzmann's formula
entropy = calculate_entropy(microstates)

# Display results
print("Microstates:", microstates)
print("Entropy:", entropy)

# Plotting the entropy over time
time = np.arange(0, time_steps)  # Time in seconds
plt.plot(time, entropy)
plt.xlabel("Time (seconds)")
plt.ylabel("Entropy")
plt.title("Entropy of the Gas Particle System over Time")
plt.tight_layout()
plt.savefig("entropy_time.png", dpi=200)
plt.show()
# %% EXAMPLE FOR A SYSTEM WITH GIVEN NUMBER OF PARTICLES

# redefine the function to calculate entropy:
def calculate_entropy_new(microstates):
    probabilities = microstates / np.sum(microstates)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Set random seed for reproducibility
np.random.seed(41)

# Define the number of particles
num_particles_list = [10, 20, 30]

# Initialize the gas particle systems
time_steps = 61  # 60 seconds + 1 (for indexing)
microstates_scaling_factor = [1, 1, 1]  # Adjust this factor to control entropy growth

# Simulate the evolution and calculate entropy for different numbers of particles
entropy_data = []
for i, num_particles in enumerate(num_particles_list):
    microstates = [np.random.randint(1, 100, num_particles)]  # Initial microstates

    # Simulate the evolution of the gas particle system
    for t in range(1, time_steps):
        # Gradually increase the number of microstates over time with scaling factor
        scaling_factor = microstates_scaling_factor[i]
        new_microstates = microstates[t-1] + scaling_factor * np.random.randint(1, 10, num_particles)
        microstates.append(new_microstates)

    # Calculate entropy using probabilities based on Boltzmann distribution
    entropy = np.zeros(time_steps)
    for t in range(time_steps):
        entropy[t] = calculate_entropy_new(microstates[t])

    entropy_data.append(entropy)

# Plotting the entropy over time for different numbers of particles
time = np.arange(0, time_steps)  # Time in seconds
for i, num_particles in enumerate(num_particles_list):
    plt.plot(time, entropy_data[i], label=f"N={num_particles}")
plt.xlabel("Time (seconds)")
plt.ylabel("Entropy")
plt.title("Entropy of Gas Particle Systems over Time")
plt.legend()
plt.tight_layout()
plt.savefig("entropy_time_N.png", dpi=200)
plt.show()
# %% ENTROPY CHANGE FOR
# define entropy change function:
def calculate_entropy_change(heat, temperature):
    entropy_change = np.cumsum(heat / temperature)
    return entropy_change

# Sample data for heat and temperature
heat = np.array([10, 15, 20, 12, 8])  # Infinitesimal heat changes
temperature = np.array([300, 310, 320, 330, 340])  # Corresponding temperatures

# Calculate entropy change
entropy_change = calculate_entropy_change(heat, temperature)

# Plotting the entropy change
plt.plot(range(len(entropy_change)), entropy_change)
plt.xlabel("Time")
plt.ylabel("Entropy Change")
plt.title("Entropy Change with Temperature Variation")
plt.tight_layout()
plt.savefig("entropy_change.png", dpi=200)
plt.show()
# %% END
