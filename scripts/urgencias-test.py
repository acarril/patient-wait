import numpy as np
import pandas as pd
import lifelines
import matplotlib.pyplot as plt

df = pd.read_csv('data/urgencias-test-noid.csv')
df


# Remove special characters from column titles
cols = df.select_dtypes(include=[np.object]).columns
df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
df = df.rename(columns={'fecha_admisión':'fecha_admision'})

# Change 0s to NaNs in fechas_
fechas = ['fecha_admision', 'fecha_triage', 'fecha_atencion', 'fecha_egreso']
df[fechas] = df[fechas].replace('0', np.nan)

# Convert columns to datetimes:
for fecha in fechas:
    df[fecha] = pd.to_datetime(df[fecha])

# Remove rows with ilogical dates:
df.drop(df[df.fecha_egreso < df.fecha_admision].index, inplace=True)
# df[df['fecha_admision'] > df['fecha_egreso']] # check

# Check too large waiting times
df['t_admin_egreso'] = df['fecha_egreso'] - df['fecha_admision']
df.sort_values(by='t_admin_egreso', ascending=False)
df = df[df['t_admin_egreso'].dt.days < 1]

df

#%% Lifelines analysis
from lifelines.utils import datetimes_to_durations
T, E = datetimes_to_durations(start_times=df.fecha_admision, end_times=df.fecha_egreso, freq='h')
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)
kmf.survival_function_
kmf.median_
kmf.plot()
plt.show()

#%% Lifelines analysis
from lifelines.utils import datetimes_to_durations
T, E = datetimes_to_durations(start_times=df.fecha_admision, end_times=df.fecha_egreso, freq='h')
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)
kmf.survival_function_
kmf.median_
kmf.plot()
plt.show()

#%% Lifelines analysis
ax = kmf.plot()
for area in df.area.unique():
    kmf.fit(T[df.area == area], E[df.area == area], label=f'Area {area}')
    if area > 1:
        kmf.plot(ax=ax)

plt.show()
