# %% IMPORTS
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
# %% MAIN
np.random.seed(0)
X = np.random.rand(1000)
Y = X + np.random.normal(0, 0.1, 1000)

def calc_mutual_information(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_XY)
    mi /= np.log(2)  # Convert from nats to bits
    return mi

bins = 30
mi = calc_mutual_information(X, Y, bins)
print(f"Mutual Information: {mi}")

# plot the scatter plot and 2D histogram:
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter plot of X and Y\nMutual Information={mi:.3f} bits')

plt.subplot(122)
plt.hist2d(X, Y, bins=bins, cmap='plasma')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'2D Histogram of X and Y\nMutual Information={mi:.3f} bits')

plt.tight_layout()
plt.savefig("mutual_information.png", dpi=200)
plt.show()


# plot the two time series:
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(X, 'b-')
plt.title('Time series of X')
plt.ylabel('Value')

plt.subplot(212)
plt.plot(Y, 'r-')
plt.title('Time series of Y')
plt.xlabel('Time')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig("mutual_information_time_series.png", dpi=200)
plt.show()

# %%
import numpy as np
from sklearn.metrics import mutual_info_score

# Generate two time series
np.random.seed(0)
t = np.arange(0, 10, 0.1)
X = np.sin(t)
Y = np.sin(t + np.random.normal(0, 1))

# Discretize the time series
X_bins = np.digitize(X, np.histogram_bin_edges(X, bins='fd'))
Y_bins = np.digitize(Y, np.histogram_bin_edges(Y, bins='fd'))

# Calculate mutual information
I = mutual_info_score(X_bins, Y_bins)

print(f"Mutual Information: {I:.2f} bits")


# %% END