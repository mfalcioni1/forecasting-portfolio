import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import ZeroInflatedPoisson

np.random.seed(42)  # for reproducibility

# Define the product hierarchy and simulation parameters
categories = ['Home', 'Clothing']
subcategories = {'Home': ['Pots', 'Pans'], 'Clothing': ['Men', 'Women']}
products = {
    'Pots': ['Pot A', 'Pot B'],
    'Pans': ['Pan A', 'Pan B'],
    'Men': ['Shirt', 'Jeans'],
    'Women': ['Dress', 'Skirt']
}

# Generate intermittent demand data
def generate_intermittent_data(n_weeks, mean_level, zero_inflation):
    """Generate zero-inflated Poisson data."""
    demand = np.random.poisson(mean_level, size=n_weeks)
    zeros = np.random.binomial(1, zero_inflation, size=n_weeks)
    demand *= (1 - zeros)
    return demand

# Create a DataFrame to hold the simulated data
demand_data = pd.DataFrame()

for category in categories:
    for subcategory in subcategories[category]:
        for product in products[subcategory]:
            data = generate_intermittent_data(24, np.random.randint(1, 10), 0.7)
            temp_df = pd.DataFrame({
                'Week': np.arange(1, 25),
                'Category': category,
                'Subcategory': subcategory,
                'Product': product,
                'Demand': data
            })
            demand_data = pd.concat([demand_data, temp_df], ignore_index=True)

def fit_zip_model(data):
    """Fit Zero-Inflated Poisson model and return fitted values."""
    X = sm.add_constant(data['Week'])  # add constant for intercept
    y = data['Demand']
    # Fit ZIP model
    zip_model = ZeroInflatedPoisson(y, X).fit(disp=0)
    return zip_model.predict(X)

def calculate_mape(actual, predicted):
    """Calculate mean absolute percentage error."""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Apply the ZIP model and calculate MAPE for each product
for product in np.unique(demand_data['Product']):
    product_data = demand_data[demand_data['Product'] == product]
    predicted_demand = fit_zip_model(product_data)
    mape = calculate_mape(product_data['Demand'], predicted_demand)
    print(f'Product: {product}, MAPE: {mape:.2f}%')
