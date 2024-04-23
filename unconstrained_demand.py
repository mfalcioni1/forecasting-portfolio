import numpy as np
import matplotlib.pyplot as plt

# Define the time periods for the two plots
weeks = np.linspace(1, 12, 12)

# Create the demand curves
# First product: normal selling season of 12 weeks
demand1 = 50 * np.exp(-0.2 * weeks)  # Exponential decay to model tapering off of interest

# Second product: sells out after 6 weeks
demand2 = 50 * np.exp(-0.2 * weeks)
demand2[weeks > 6] = 0  # Set demand to zero after week 6 due to stocking out

# Create the plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for the first product
axes[0].plot(weeks, demand1, marker='o', color='blue')
axes[0].set_title('Demand Curve for Product 1')
axes[0].set_xlabel('Weeks')
axes[0].set_ylabel('Demand')
axes[0].grid(True)

# Plot for the second product
axes[1].plot(weeks, demand2, marker='o', color='red')
axes[1].set_title('Demand Curve for Product 2')
axes[1].set_xlabel('Weeks')
axes[1].set_ylabel('Demand')
axes[1].grid(True)

# Display the plots
plt.tight_layout()
plt.show()
# save the plots to /images
plt.savefig('images/unconstrained_demand.png')

# Impute the demand for Product 2 using the pattern from Product 1 for weeks 7 to 12
imputed_demand2 = demand2.copy()
imputed_demand2[weeks > 6] = demand1[weeks > 6]

# Plot the original and imputed demand curves for Product 2
plt.figure(figsize=(10, 6))
plt.plot(weeks, demand2, marker='o', color='red', label='Original Demand for Product 2')
plt.plot(weeks, imputed_demand2, marker='x', linestyle='--', color='green', label='Imputed Demand for Product 2')
plt.title('Original vs. Imputed Demand for Product 2')
plt.xlabel('Weeks')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('images/imputed_demand.png')

# Intermittent Demand
# Define the mean demand rate for each week
mean_demand = np.array([3, 2, 2, 1, 0, 2, 0, 1, 1, 1, 0, 0])

# Generate random Poisson-distributed demand values for each week
#np.random.seed(42)  # For reproducibility
store_demand = np.random.poisson(mean_demand, size=12)

# Plotting the simulated demand
plt.figure(figsize=(10, 6))
plt.bar(weeks, store_demand, color='purple')
plt.title('Simulated Weekly Demand at a Store (Poisson Model)')
plt.xlabel('Weeks')
plt.ylabel('Demand')
plt.grid(True)
plt.show()
