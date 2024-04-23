import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

np.random.seed(42)
n_weeks = 24
n_products = 3
data = {
    'product_id': ['Product1', 'Product2', 'Product3'],
    'demand': [np.random.poisson(lam=20, size=n_weeks) for _ in range(n_products)],
    'category': ['Electronics', 'Electronics', 'Home']
}

start_date = pd.Timestamp("2021-01-01", freq='W')

training_data = ListDataset(
    [{
        FieldName.TARGET: data['demand'][i],
        FieldName.START: start_date,
        FieldName.ITEM_ID: data['product_id'][i],
        FieldName.FEAT_STATIC_CAT: [i]  # Assume category encoded as static feature
    } for i in range(n_products)],
    freq='W'
)

estimator = DeepVARHierarchicalEstimator(
    freq="W",
    prediction_length=4,
    trainer=Trainer(epochs=10, learning_rate=1e-3),
    use_feat_static_cat=True,
    cardinality=[len(set(data['category']))],
    num_layers=2,
    num_cells=40
)

predictor = estimator.train(training_data=training_data)

test_data = ListDataset(
    [{
        FieldName.TARGET: np.zeros(n_weeks),
        FieldName.START: start_date,
        FieldName.ITEM_ID: 'NewProduct',
        FieldName.FEAT_STATIC_CAT: [1]  # Example: New electronics product
    }],
    freq='W'
)


forecast_it, ts_it = make_evaluation_predictions(dataset=test_data, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)
tss = list(ts_it)

# Visualize the forecast
import matplotlib.pyplot as plt

for test_entry, forecast in zip(tss, forecasts):
    plt.figure(figsize=(10, 6))
    plt.plot(pd.date_range(start_date, periods=len(test_entry['target']), freq='W'), test_entry['target'], label='Historical Demand')
    plt.plot(pd.date_range(start_date, periods=len(forecast.mean), freq='W')[-4:], forecast.mean, label='Forecasted Demand')
    plt.legend()
    plt.show()
