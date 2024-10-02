# %%
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima import model
import matplotlib.pyplot as plt

# %%
file_path = r"C:\Users\trucn\Documents\repositories\ds-mod-2-0924-code-solutions\time-series-2\AirPassengers.csv"
df = pd.read_csv(file_path, parse_dates = ['Month'], index_col = 'Month')

# %%
# Rename #Passengers to Pass for simplicity

df = df.rename(columns = {'#Passengers':'Pass'})

# %%
# Save a copy of df for future analyses
df1 = df.copy()

# %% [markdown]
# # 1. Use “plot_pacf” and “plot_acf” to get the “p” and “q” values respectively.

# %%
plot_pacf(df['Pass'].dropna());

# %% [markdown]
# Based on the PACF plot, since there's a sudden dropoff, p = 1.

# %%
plot_acf(df['Pass'].dropna());

# %% [markdown]
# Based on the ACF plot, since there seems to be some hilliness, we can't precisely determine q.

# %% [markdown]
# Let's try taking the level 2 difference.

# %%
df['PassDiff'] = df['Pass'].diff(2)

# %%
plot_pacf(df['PassDiff'].dropna()); 

# %% [markdown]
# p = 1

# %%
plot_acf(df['PassDiff'].dropna());

# %% [markdown]
# q = 1

# %% [markdown]
# Now I want to see if p and q stay the same if we take the logarithmic transformation of the data.

# %%
df['PassLog'] = np.log(df['Pass'])

# %%
plot_pacf(df['PassLog'].dropna());

# %% [markdown]
# p = 1

# %%
plot_acf(df['PassLog'].dropna());

# %% [markdown]
# Cannot precisely determine q.

# %% [markdown]
# # 2. Build an ARIMA model based on the “p” and “q” values obtained from above and get the RMSE.

# %%
ts = df[['Pass']]

# %%
m = model.ARIMA(ts, order = (1, 2, 1)) # p = 1, d = 2, q = 1
results_pass = m.fit()

# %%
results_pass.arparams

# %%
results_pass.maparams

# %%
results_pass.fittedvalues

# %%
plt.plot(ts, color = 'navy')
plt.plot(results_pass.fittedvalues, color = 'aqua')

# %%
mean_squared_error(ts, results_pass.fittedvalues.dropna(), squared = False)

# %% [markdown]
# RMSE ≈ 33.72


