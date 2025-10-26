import pandas as pd

df = pd.read_csv(
    "-t_tempat_pelayanan_kesehatan_dinkes_2018-2024.csv",
    delimiter=";"
)

df.columns = df.columns.str.strip()

puskesmas_time_series = df.groupby('Tahun')['Puskesmas'].sum().reset_index()

puskesmas_time_series['Tahun'] = pd.to_datetime(puskesmas_time_series['Tahun'], format='%Y')
puskesmas_time_series.set_index('Tahun', inplace=True)

print("Data Puskesmas per Tahun (2018-2024):")
print(puskesmas_time_series)

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

model = sm.tsa.statespace.SARIMAX(
    puskesmas_time_series['Puskesmas'],
    order=(0, 0, 0),          # p=0, d=0, q=0 (Karena datanya konstan)
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
# print(results.summary()) # Untuk melihat ringkasan hasil model

forecast_steps = 3
pred_sarima = results.get_prediction(start=len(puskesmas_time_series), end=len(puskesmas_time_series) + forecast_steps - 1)
pred_ci = pred_sarima.conf_int() # Interval kepercayaan

forecast_index = pd.to_datetime(
    [str(2024 + i) for i in range(1, forecast_steps + 1)],
    format='%Y'
)
forecast_values = pd.Series(pred_sarima.predicted_mean.values, index=forecast_index)

print("\nPrediksi Total Puskesmas (2025-2027):")
print(forecast_values)