# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import statistics
import category_encoders as ce

# %%
pd.set_option('display.max_columns', 200)

# %%
df = pd.read_csv(r'C:\Users\trucn\Documents\repositories\ds-mod-2-0924-code-solutions\clustering\clustering\imports-85.data', header = None)

# %%
df

# %%
# You should only input the column names that you need, not all of them!

# %% [markdown]
# ## 1. Using 'price' and 'horsepower' columns from import-85.data, generate a KMeans model.

# %% [markdown]
# #### Find the optimal value of k using elbow method, silhouette score, and calinski-harabasz score.

# %%
# Price is column #25 and horsepower is column #21

# %% [markdown]
# #### Elbow method

# %%
#price_no_quests = df.drop(df[df[25] == '?'].index, inplace = True)

# %%
price_no_quests = df[df[25] != '?']

# %%
price_no_quests[25].dtype # Turns out price is an object type

# %%
price_no_quests[25] = price_no_quests[25].astype(float)

# %%
price_avg = round(statistics.mean(price_no_quests[25]))

# %%
df[25] = df[25].apply(lambda x: price_avg if x == '?' else x)

# %%
df[25] = df[25].astype(float)

# %%
horsepower_no_quests = df[df[21] != '?']

# %%
horsepower_no_quests[21] = horsepower_no_quests[21].astype(float)

# %%
horsepower_avg = round(statistics.mean(horsepower_no_quests[21]))

# %%
df[21] = df[21].apply(lambda x: horsepower_avg if x == '?' else x)

# %%
df[21] = df[21].astype(float)

# %%
X = df[[25, 21]]

# %%
plt.scatter(df[25], df[21])

# %%
inertia = []
s_score = [] # Silhouette: Maximize this as close to 1
c_score = [] # Calinski-Harabasz: Maximize whichever's highest

for k in range (2, 11): # For k values between 2 and 10
    km = KMeans(n_clusters = k, n_init = 20)
    km.fit(X)
    inertia.append(km.inertia_)
    s_score.append(silhouette_score(X, km.labels_))
    c_score.append(calinski_harabasz_score(X, km.labels_))

# %%
inertia

# %%
s_score

# %%
c_score

# %%
plt.plot(range(2, 11), inertia)

# %%
plt.plot(range(2, 11), s_score)

# %%
plt.plot(range(2, 11), c_score)

# %%
# Let's try k = 3

model = KMeans(n_clusters = 3)
model.fit_transform(X)

# %%
plt.scatter(X[25], X[21], c = model.labels_)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker = 'x', c = 'r')

# %%
mms = MinMaxScaler()
X_scaled = pd.DataFrame(mms.fit_transform(X), columns = X.columns)
X_scaled

# %%
inertia_scaled = []
s_score_scaled = [] # Silhouette: Maximize this as close to 1
c_score_scaled = [] # Calinski-Harabasz: Maximize whichever's highest

for k in range (2, 11): # For k values between 2 and 10
    km_scaled = KMeans(n_clusters = k, n_init = 20)
    km_scaled.fit(X_scaled)
    inertia_scaled.append(km_scaled.inertia_)
    s_score_scaled.append(silhouette_score(X_scaled, km_scaled.labels_))
    c_score_scaled.append(calinski_harabasz_score(X_scaled, km_scaled.labels_))

# %%
model_scaled = KMeans(n_clusters = 3)
model_scaled.fit_transform(X_scaled)

# %%
plt.scatter(X_scaled[25], X_scaled[21], c = model_scaled.labels_)
plt.scatter(model_scaled.cluster_centers_[:,0], model_scaled.cluster_centers_[:,1], marker = 'x', c = 'r')

# %% [markdown]
# # 2. Using 'price' and 'horsepower' columns from import-85.data, generate a DBSCAN model.

# %% [markdown]
# #### Plot your results to visualize outliers.

# %%
dbs = DBSCAN(eps = 0.1057825, min_samples = 7)
dbs.fit(X_scaled)

# %%
plt.scatter(X_scaled[25], X_scaled[21], c = dbs.labels_)

# %%
dbs.labels_

# %%
labs = dbs.labels_

# %%
labs[dbs.core_sample_indices_] = 1

# %%
labs

# %%
plt.scatter(X_scaled[25], X_scaled[21], c = labs)

# %% [markdown]
# # 3. Using mlb_batting_cleaned.csv, write a function that takes a player's name and shows the 2 closest players using the nearest neighbors algorithm.

# %%
mlb = pd.read_csv(r'C:\Users\trucn\Documents\repositories\ds-mod-2-0924-code-solutions\clustering\clustering\mlb_batting_cleaned.csv')

# %%
# Try utilizing np.where somewhere in thi

mlb

# %%
mlb2 = mlb.copy()

# %%
# OHE Tm column
ohe = ce.OneHotEncoder(use_cat_names = True)
ohe.fit(mlb2['Tm'])
tm_ohe = ohe.transform(mlb2['Tm'])

# %%
# OHE Lg column
ohe.fit(mlb2['Lg'])
lg_ohe = ohe.transform(mlb2['Lg'])

# %%
# Concat the newly-encoded columns into original dataframe

mlb2 = pd.concat([mlb2, tm_ohe, lg_ohe], axis = 1)

# %%
# Drop the two original columns that were encoded
mlb2 = mlb2.drop(columns = ['Tm', 'Lg'])

# %%
# Drop Name
mlb2 = mlb2.drop(columns = 'Name')

# %%
# Min-max scale all the columns
scaler = MinMaxScaler()
mlb2 = pd.DataFrame(scaler.fit_transform(mlb2), columns = mlb2.columns)

# %%
nn = NearestNeighbors(n_neighbors = 3)
nn.fit(mlb2)

# %%
dist, indx = nn.kneighbors(mlb2)

# %%
dist

# %%
indx

# %%
mlb.loc[mlb['Name'] == 'Shohei Ohtani'] # index 524

# %%
indx[524]

# %%
mlb.iloc[726,:]

# %%
mlb.iloc[765,:]

# %% [markdown]
# # Function that retrieves the two closest players

# %%
def nearest_two_players(data, player_name):
    
    # Get the list of all player names saved separately and the index of the specific player inputted
    player_names_list = data['Name']
    player_name_index = data[data['Name'] == player_name].index.values.astype(int)[0]
    
    # OHE Tm and Lg
    ohe = ce.OneHotEncoder(use_cat_names = True)
    ohe.fit(data['Tm'])
    tm_ohe = ohe.transform(data['Tm'])

    ohe.fit(data['Lg'])
    lg_ohe = ohe.transform(data['Lg'])

    # Concat the encoded Tm and Lg columns to original dataframe
    data = pd.concat([data, tm_ohe, lg_ohe], axis = 1)
    
    # Drop Name, Tm, and Lg
    data = data.drop(columns = ['Name', 'Tm', 'Lg'])

    # Min-max scale the columns
    scal = MinMaxScaler()
    data = pd.DataFrame(scal.fit_transform(data), columns = data.columns)
    
    # Build the model
    nn = NearestNeighbors(n_neighbors = 3)
    nn.fit(data)
    
    # Get dist and indx
    dist, indx = nn.kneighbors(data)
    
    # Index 0 is the original player's name, index 1 is the first match, and index 2 is the second match
    two_matches = indx[player_name_index]

    print(f"Input player name: {player_names_list[player_name_index]}")
    print(f"The first closest player: {player_names_list[two_matches[1]]}")
    print(f"The second closest player: {player_names_list[two_matches[2]]}")
    

# %%
nearest_two_players(mlb, 'Shohei Ohtani')

# %% [markdown]
# # Testing code to make function

# %%
player_name_index  = mlb[mlb['Name'] == 'Shohei Ohtani'].index.values.astype(int)[0]

# %%
two_matches = indx[player_name_index]

# %%
two_matches[0]

# %%
two_matches[1]

# %%
two_matches[2]

# %%
player_names_list = mlb['Name']

# %%
player_names_list.iloc[two_matches[2]]


