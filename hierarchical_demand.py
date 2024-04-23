import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.random.seed(42)

weeks = np.arange(1, 25)
products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 'Product F', 'Product G', 'New Product']
categories = ['Category 1', 'Category 1', 'Category 1', 'Category 1', 'Category 2', 'Category 2', 'Category 2',  'Category 2']
demand = np.random.poisson(lam=[3, 2, 4, 3, 2, 4, 2, 0], size=(24, len(products))) * (np.random.rand(24, len(products)) < 0.3)

data = pd.DataFrame(demand, columns=products, index=pd.date_range("20210101", periods=24, freq='W'))
data['Week'] = data.index

data_long = pd.melt(data.reset_index(), id_vars=['index', 'Week'], value_vars=products, var_name='Product', value_name='Demand')
data_long['Category'] = data_long['Product'].map(dict(zip(products, categories)))

category_demand = data_long.groupby(['Week', 'Category'])['Demand'].sum().reset_index()
category_demand['Week'] = pd.to_datetime(category_demand['Week'])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

forecasts = {}
for category in category_demand['Category'].unique():
    category_data = category_demand[category_demand['Category'] == category].set_index('Week')
    model = ExponentialSmoothing(category_data['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
    future_index = pd.date_range(category_data.index[-1] + pd.Timedelta(weeks=1), periods=12, freq='W')
    forecast = model.forecast(12)
    forecasts[category] = forecast

    axes[0].plot(category_data.index, category_data['Demand'], label=f'Actual Demand for {category}')
    axes[0].plot(future_index, forecast, linestyle='--', label=f'Forecast for {category}')

axes[0].set_title('Aggregated Category Demand & Forecasts')
axes[0].set_ylabel('Demand')
axes[0].legend()

# Assuming equal market share for new products
new_product_forecast = forecasts['Category 2'] / 4
axes[1].plot(future_index, new_product_forecast, color='red', label='Forecast for New Product')
axes[1].set_title('Forecast for New Product in Category 2')
axes[1].set_ylabel('Demand')
axes[1].set_xlabel('Weeks')
axes[1].legend()

plt.tight_layout()
plt.show()
