"""
A script to demonstrate the concept of entropy.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: Feb 02, 2023

For reproducibility:

```powershell
conda create -n entropy -y python=3.9
conda activate entropy
conda install -y mamba
mamba install -y ipykernel numpy matplotlib scipy
```
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# %% ENTROPY AS A FUNCTION OF MICROSTATES
import numpy as np
import matplotlib.pyplot as plt

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
# %% ENTROPY FOR TWO CLOSED SYSTEMS
# re-defining the entropy function:
def entropy(probs):
    return -np.sum(probs * np.log(probs))

# Set random seed for reproducibility
np.random.seed(42)

# Number of particles
n = 10

# Generate particle positions for first box
x1 = np.random.normal(loc=0.5, scale=0.1, size=n)
y1 = np.random.normal(loc=0.5, scale=0.1, size=n)

# Generate particle positions for second box
x2 = np.random.uniform(size=n)
y2 = np.random.uniform(size=n)

# Calculate probability density for first box
kde1 = gaussian_kde(np.vstack([x1, y1]))
xgrid, ygrid = np.mgrid[0:1:100j, 0:1:100j]
probs1 = kde1(np.vstack([xgrid.ravel(), ygrid.ravel()]))
probs1 /= probs1.sum()

# Calculate probability density for second box
kde2 = gaussian_kde(np.vstack([x2, y2]))
probs2 = kde2(np.vstack([xgrid.ravel(), ygrid.ravel()]))
probs2 /= probs2.sum()

# Calculate entropy for each box
entropy1 = entropy(probs1)
entropy2 = entropy(probs2)

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=2)

# Plot particle positions and probability density for first box
ax1.scatter(x1, y1)
im1 = ax1.imshow(np.fliplr(probs1.reshape(xgrid.shape)), origin='upper', extent=[0, 1, 0, 1])
ax1.set_title(f'System 1: Entropy = {entropy1:.3f}')
cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', label='Probability Density')
cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=90)
ticks = np.linspace(im1.get_clim()[0], im1.get_clim()[1], 4)
cbar1.set_ticks(ticks)

# Plot particle positions and probability density for second box
ax2.scatter(x2, y2)
im2 = ax2.imshow(np.fliplr(probs2.reshape(xgrid.shape)), origin='upper', extent=[0, 1, 0, 1])
ax2.set_title(f'System 2: Entropy = {entropy2:.3f}')
cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', label='Probability Density')
cbar2.ax.set_xticklabels(cbar2.ax.get_xticklabels(), rotation=90)
ticks = np.linspace(im2.get_clim()[0], im2.get_clim()[1], 4)
cbar2.set_ticks(ticks)
plt.tight_layout()
plt.savefig("entropy_two_systems.png", dpi=200)
plt.show()
# %% ENTROPY FOR DIFFERENT NUMBER OF PARTICLES AS A FUNCTION OF TIME

def calculate_entropy(particles):
    """Calculate the entropy of a system of particles."""
    # Calculate the histogram of particle positions
    hist, _ = np.histogram(particles, bins=10, range=(0, 1), density=True)
    
    # Calculate the probabilities
    probs = hist / hist.sum()
    
    # Calculate the entropy
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    
    return entropy

# Set random seed for reproducibility
np.random.seed(42)

# Number of time steps
T = 100

# Number of particles
N_values = [10, 25, 200]

# Create figure
fig, ax = plt.subplots()

# Loop over number of particles
for N in N_values:
    # Initialize particle positions
    particles = np.random.uniform(size=N)

    # Initialize entropy array
    S = np.zeros(T)

    # Loop over time steps
    for t in range(T):
        # Update particle positions
        particles += 0.01 * np.random.randn(N)

        # Calculate entropy
        S[t] = calculate_entropy(particles)

    # Plot entropy vs time
    ax.plot(S, label=f'N={N}')

# Add legend and labels
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Entropy')
plt.tight_layout()
plt.savefig("entropy_different_number_particles.png", dpi=200)
plt.show()
# %% INCREASING ENTROPY FOR DIFFERENT NUMBER OF PARTICLES AS A FUNCTION OF TIME

def entropy(probs):
    return -np.sum(probs * np.log(probs))

# Set random seed for reproducibility
np.random.seed(42)

# Number of time steps
T = 100

# Number of particles
N_values = [10, 20, 100]

# Create figure
fig, ax = plt.subplots()

# Loop over number of particles
for N in N_values:
    # Initialize particle positions
    x = np.random.normal(loc=0.5, scale=0.1, size=N)
    y = np.random.normal(loc=0.5, scale=0.1, size=N)

    # Initialize entropy array
    S = np.zeros(T)

    # Loop over time steps
    for t in range(T):
        # Update particle positions
        x += 0.1 * np.random.randn(N)
        y += 0.1 * np.random.randn(N)

        # Calculate probability density
        kde = gaussian_kde(np.vstack([x, y]))
        xgrid, ygrid = np.mgrid[0:1:100j, 0:1:100j]
        probs = kde(np.vstack([xgrid.ravel(), ygrid.ravel()]))
        probs /= probs.sum()

        # Calculate entropy
        S[t] = entropy(probs)

    # Plot entropy vs time
    ax.plot(S, label=f'N={N}')

# Add legend and labels
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Entropy')

plt.show()
# %% END
