# Adjusted simulation - increase cluster variation for non-zero ICC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Function to compute logistic growth using natural log in the exponent
def logistic_growth(t, r, A):
    return 1 / (1 + np.exp(-(r * t + np.log(A))))

# Parameters
num_clusters = 10
users_per_cluster = 50  # Reduced for efficiency, but captures essence
time_points = np.array([0, 5, 10, 15, 20])
base_r = 0.2
base_M0 = 0.01

# Introduce cluster-level variation in growth rate
cluster_rs = base_r + np.random.normal(0, 0.15, num_clusters)  # Larger sd for detectable clustering
data = []

# Generate data: Loop over clusters and users
for cluster in range(num_clusters):
    r = cluster_rs[cluster]
    for user in range(users_per_cluster):
        # User-specific initial condition with variation
        A = base_M0 / (1 - base_M0) * np.exp(np.random.normal(0, 1))  # Larger sd for diversity
        M = logistic_growth(time_points, r, A)
        M += np.random.normal(0, 0.01, len(time_points))  # Reduced noise for clearer patterns
        M = np.clip(M, 1e-5, 1 - 1e-5)  # Prevent log issues
        for tp, m in zip(time_points, M):
            data.append({'Cluster': cluster, 'User': user, 'Time': tp, 'Adoption': m})

# Create DataFrame
df = pd.DataFrame(data)

# Compute logit transform using natural log
df['Logit'] = np.log(df['Adoption'] / (1 - df['Adoption']))

# Fit mixed linear model: Logit ~ Time (fixed) + (1|Cluster) for random intercept
md = sm.MixedLM.from_formula("Logit ~ Time", data=df, groups=df["Cluster"])
mdf = md.fit(method='lbfgs')  # Optimizer for convergence

# Extract variance components and compute ICC
cluster_var = mdf.cov_re.iloc[0, 0]
residual_var = mdf.scale
icc = cluster_var / (cluster_var + residual_var)

print(f"Simulated ICC: {icc}")

# Print model summary for details
print(mdf.summary())

# Plot mean adoption growth for two clusters to illustrate differences
plt.figure()
for cluster in [0, 1]:
    cluster_data = df[df['Cluster'] == cluster]
    mean_adoption = cluster_data.groupby('Time')['Adoption'].mean()
    plt.plot(mean_adoption.index, mean_adoption.values, label=f'Cluster {cluster} Mean')
plt.title('Mean Meme Adoption Growth in Two Clusters')
plt.xlabel('Time (days)')
plt.ylabel('Adoption Proportion')
plt.legend()
plt.show()

# Data summary
print(df.describe())