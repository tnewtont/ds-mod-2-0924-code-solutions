# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.arima import model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import kruskal 
from statsmodels.tsa.seasonal import STL
import yfinance as yf
from scipy.signal import periodogram
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import sqlite3

# %%
# 1 year of data in 1 hour intervals

# %%
rl = yf.Ticker('RL') # Ralph Lauren Corporation

# %%
# ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

# %%
rl_data = yf.download('RL', period = '1y', interval = '60m')
rl_data.to_csv('rl_data.csv')

# %%
rl_df = pd.read_csv('rl_data.csv')
rl_df

# %%
rl_df.info()

# %%
rl_df['Datetime'].dtype

# %%
rl_df = rl_df[['Datetime', 'Open']]

# %%
rl_df['Datetime'] = pd.to_datetime(rl_df['Datetime'])

# %%
rl_df.set_index('Datetime', inplace = True)

# %%
rl_df.sort_index(inplace = True)

# %%
rl_df_diff1 = rl_df['Open'].diff(1)

# %%
rl_df_diff2 = rl_df['Open'].diff(2)

# %%
adfuller(rl_df['Open'])

# %%
adfuller(rl_df_diff1.dropna()) # d = 1

# %%
# Use a factor of 7 (7 hours per day since that's how long the stock market is open for, and it's 5 days a week)

# %%
# Most seasonal period function (if it is seasonal which period most likely)

# %%
def find_period(signal):
    acf = np.correlate(signal, signal, 'full')[-len(signal):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    return peaks[acf[peaks].argmax()]

# %%
rl_df.shape

# %%
rl_df[['Open']]

# %%
open_vals = rl_df['Open'].reset_index()

# %%
open_vals['Open']

# %%
plot_acf(rl_df_diff1.dropna(), lags = np.arange(0, 600, 7)); # q = 1 # look at first 600 lags, every 7
plt.ylim(-0.1,0.1)

# %%
plot_pacf(rl_df_diff1.dropna()); # p = 1

# Do ARIMA if data has no seasonality

# %%
rl_group = [rl_df.iloc[i::7] for i in range(7)] # 7 because it's 7 hours each day
rl_group

# %%
kruskal(*rl_group) # p-value is basically 1, so there's insufficient evidence for seasonality.

# %%
# Instead of getting hourly data, you might want to get minutely 
# 60 mins * number of days for each month * 7 hours

# %%
p = list(range(1,4))
d = list(range(0,3))
q = list(range(0,3))

pdq_list = list(itertools.product(p,d,q))

# %%
len(rl_df)

# %%
len(rl_df) * 0.75

# %%
len(rl_df) - 1317 #439

# %%
0.75 * len(rl_df)

# %%
train = rl_df[:1317]
test = rl_df[1317:]
rmse = []

# %% [markdown]
# # From the original data, store it into a list, and x-values to np.linspace

# %%
for i in tqdm(range(len(pdq_list))):
    m = model.ARIMA(train, order = pdq_list[i])
    fitted = m.fit()
    preds = fitted.forecast(439)
    rmse.append(mean_squared_error(test, preds, squared = False))

# %%
np.argmin(rmse) # 20

# %%
rmse[20]

# %%
pdq_list[20] 

# %%
best_model = model.ARIMA(rl_df, order = pdq_list[20])
results = best_model.fit()

# %%
1318 + 439

# %%
1758 + 439

# %%
# We have to convert the timestamps to numerical integers since Python's having problems with determining 
# the future timestamps beyond today, for some reason. This is also to make plotting the data possible

ls1 = np.arange(1,1757, 1) # Timestamps of the original data converted to a series of subsequent integers (1757 is non-inclusive)
ls2 = np.arange(1757, 2196, 1) # Would-be timestamps of the forecasted values
original_vals = rl_df['Open'].values # Original values stored separately

# %%
predicted_vals = results.fittedvalues.values

# I did .values at the end since I want to exclude the timestamps, otherwise
# we run into an issue of trying to plot out the values

# %%
flist = results.forecast(439).values

# %%
len(ls1)

# %%
len(ls2)

# %%
len(original_vals)

# %%
len(flist)

# %%
plt.plot(ls1, original_vals, color = 'aqua')
plt.plot(ls1, predicted_vals, color = 'red')
plt.plot(ls2, flist, color = 'navy')

# %%
# That particular frequency (peak) has a higher magnitude, so that frequency contributes significantly

# %%
# plt.plot(rl_df, color = 'blue')
# plt.plot(results.fittedvalues, color = 'yellow')

# %%
# I decided to plot a periodogram in order to further investigate any seasonality in the data
f, Pxx_den = periodogram(rl_df['Open']) # Anything above two times above the original frequency

# %%
plt.figure()
plt.semilogy(f[1:], Pxx_den[1:]) # Semi logarithmic y
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V**2/Hz]')
plt.title('Periodogram')
plt.show

# %%
data_array = np.array([ls1, predicted_vals, original_vals])

# %%
data_array.shape

# %%
database = pd.DataFrame(data_array.T)

# %%
database.rename(columns = {0:'timestamp', 1: 'prediction', 2: 'actual_values'}, inplace = True)

# %%
database

# %%
# Using SQLite to save this dataframe into a database
conn = sqlite3.connect('rl.db')
database.to_sql('rl_data', conn, if_exists = 'replace', index = False)

pd.read_sql('SELECT * FROM rl_data', conn)



# %%
# What types of stocks display seasonality?

# Exogenous factors can affect the model

# Random forest regressor, theta predictor, prophet 

# San Francisco traffic

# Fast Fourier Transform the model too


