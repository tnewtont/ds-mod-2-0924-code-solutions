# %% [markdown]
# # 1. Plot the time series data with rolling mean and rolling standard deviation and see if it is stationary.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# Logarithmic, reciprocal, and square root

# %%
df = pd.read_csv(r"C:\Users\trucn\Documents\repositories\ds-mod-2-0924-code-solutions\time-series-1\AirPassengers.csv")

# %%
pd.set_option("display.max_columns", 250)

# %%
df

# %%
# First set Month as DateTime type and as the index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace = True)


# %%
df.sort_index

# %%
df1 = df.copy() # Save a copy of the tidied data for future analyses

# %%
df['Rolling Mean'] = df.rolling(window = 12).mean()

# %%
df['Rolling Std'] = df['#Passengers'].rolling(window = 12).std()

# %%
plt.plot(df['#Passengers'], color = 'navy')
plt.plot(df['Rolling Mean'], color = 'deepskyblue')
plt.plot(df['Rolling Std'], color = 'darkorange')

# %% [markdown]
# The data appears to be not stationary, particularly due to the increasing mean and variance.

# %% [markdown]
# # 2. Try different levels of differences, and plot the time series data with rolling mean and standard deviation. See if it is stationary.

# %% [markdown]
# ## Level of difference, d = 1

# %%
#df['#Passengers1'] = df['#Passengers'].shift()
#df['Diff1'] = df['#Passengers'] - df['#Passengers1']

# %%
df['#Passengers_Diff1'] = df['#Passengers'].diff(1)

# %%
df['Rolling Mean Diff1'] = df['#Passengers_Diff1'].rolling(window = 12).mean()

# %%
df['Rolling Std Diff1'] = df['#Passengers_Diff1'].rolling(window = 12).std()

# %%
plt.plot(df['#Passengers_Diff1'], color = 'navy')
plt.plot(df['Rolling Mean Diff1'], color = 'deepskyblue')
plt.plot(df['Rolling Std Diff1'], color = 'darkorange')

# %% [markdown]
# At d = 1, the data is not stationary.

# %% [markdown]
# ## Level of difference, d = 2

# %%
df['#Passengers_Diff2'] = df['#Passengers'].diff(2)
df['Rolling Mean Diff2'] = df['#Passengers_Diff2'].rolling(window = 12).mean()
df['Rolling Std Diff2'] = df['#Passengers_Diff2'].rolling(window = 12).std()

# %%
plt.plot(df['#Passengers_Diff2'], color = 'navy')
plt.plot(df['Rolling Mean Diff2'], color = 'deepskyblue')
plt.plot(df['Rolling Std Diff2'], color = 'darkorange')

# %%
plt.plot(df['#Passengers_Diff1'], color = 'navy')
plt.plot(df['Rolling Mean Diff1'], color = 'deepskyblue')
plt.plot(df['Rolling Std Diff1'], color = 'darkorange')

# %% [markdown]
# At d = 2, data still looks non-stationary.

# %% [markdown]
# # 3. Try to transform the data, and make different levels of differences. See if it is stationary.

# %%
df1['#Passengers'].hist()

# %%
# Let's try taking the logarithm of #Passengers since it appears right-skewed

# %%
df1['#PassengersLog'] = np.log(df1['#Passengers'])

# %% [markdown]
# ## Level of difference, d = 1

# %%
df1['#Passengers_LogDiff1'] = df1['#PassengersLog'].diff(1)

# %%
df1['#Passengers_LogDiff1'].dropna().plot()

# %% [markdown]
# ## Level of difference, d = 2

# %%
df1['#Passengers_LogDiff2'] = df1['#PassengersLog'].diff(2)

# %%
df1['#Passengers_LogDiff2'].plot()

# %% [markdown]
# # 4. Get the p-value from Augmented Dickey-Fuller test to make the data stationary.

# %%
adf_original_data = adfuller(df['#Passengers'].dropna())
adf_original_data 

# In this context, if we fail to reject the null hypothesis, we cannot conclude that our data is stationary.

# %%
adf_original_data2 = adfuller(df['#Passengers_Diff1'].dropna())
adf_original_data2 

# %%
adf_original_data3 = adfuller(df['#Passengers_Diff2'].dropna())
adf_original_data3 

# %% [markdown]
# For the original data, at a level of difference 2, we are able to reject the null hypothesis at the 5% and 10% significance levels and conclude that the data is stationary.

# %%
adf_trans_data = adfuller(df1['#PassengersLog'].dropna())
adf_trans_data

# %%
adf_trans_data2 = adfuller(df1['#Passengers_LogDiff1'].dropna())
adf_trans_data2

# %%
adf_trans_data3 = adfuller(df1['#Passengers_LogDiff2'].dropna())
adf_trans_data3

# %% [markdown]
# For the transformed data, at a level of difference 2, we are able to reject the null hypothesis at the 5% and 10% significance levels and conclude that the data is stationary.


