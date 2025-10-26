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