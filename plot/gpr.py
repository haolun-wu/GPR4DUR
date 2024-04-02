import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set theme and seed
sns.set_theme(style="whitegrid")
np.random.seed(43)  # seed for reproducibility


# Define a kernel function
def kernel(x, y, l=1.0, sigma_f=1.0):
    """Isotropic squared exponential kernel."""
    sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(y ** 2, 1) - 2 * np.dot(x, y.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


# Create an array of test points
X = np.linspace(-5, 5, 100).reshape(-1, 1)

# Ground truth
true_y = np.sin(X).ravel()

# Simulated observations
X_obs = np.sort(np.random.choice(X.ravel(), size=7, replace=False)).reshape(-1, 1)
y_obs = np.sin(X_obs).ravel() + np.random.normal(0, 0.1, size=X_obs.shape[0])  # slight noise added

# Compute the mean and covariance of the posterior distribution
K_obs = kernel(X_obs, X_obs)
K_obs_inv = np.linalg.inv(K_obs + 0.1 ** 2 * np.eye(len(X_obs)))  # Adding a bit of noise for stability
K_cross = kernel(X, X_obs)
K = kernel(X, X)

mu_post = K_cross @ K_obs_inv @ y_obs
cov_post = K - K_cross @ K_obs_inv @ K_cross.T

# Generate samples from the posterior at our test points
samples = np.random.multivariate_normal(mu_post, cov_post, 2)

# Fontsize parameters
legend_fontsize = 14
tick_label_fontsize = 14  # adjust this for x and y axis tick label sizes

# Plot
plt.figure(figsize=(9, 5))
plt.plot(X, true_y, 'r', label='True Function')
plt.scatter(X_obs, y_obs, c='black', marker='x', s=50, zorder=10, label='Observations')

# Adding the line for the posterior mean
plt.plot(X, mu_post, c='black', linestyle='dashdot', lw=1, label='Posterior Mean')

for i, sample in enumerate(samples):
    plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')

# Uncertainty region
plt.fill_between(X.ravel(), mu_post - 1.96 * np.sqrt(np.diag(cov_post)),
                 mu_post + 1.96 * np.sqrt(np.diag(cov_post)), alpha=0.2, label='95% Confidence Interval')

plt.xlabel('X', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend(fontsize=legend_fontsize, loc='lower left')

# Adjust tick label sizes
plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
plt.tight_layout()
plt.savefig('../save_figure/gpr.png')

plt.show()
