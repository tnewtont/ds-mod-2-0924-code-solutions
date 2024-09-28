# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%
df = pd.read_csv(r'C:\Users\trucn\Documents\repositories\ds-mod1-0724-code-solutions\dimensionality-reduction\mnist.csv')

# %%
df.isna().sum().sum() # Verifying there are no nulls to begin with

# %%
df.columns

# %%
len(df.columns) # There are 785 features

# %%
pca = PCA(n_components = 785)
df_trsf = pca.fit_transform(df)

# %%
pca.explained_variance_ratio_

# %%
num_components = 0
sum = 0
for r in pca.explained_variance_ratio_:
    num_components += 1
    sum += r
    if sum > 0.9:
        break

# %%
sum # Cumulative sum is more-or-less close to 0.9

# %%
num_components # 84 components

# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# %%
# Let's construct a PCA model with n_constructors = 84
pca84 = PCA(n_components = 84)
df_trsf_84 = pca84.fit_transform(df)

# %%
# Obtain the original values
pca84.inverse_transform(df_trsf_84)

# %%
plt.plot(np.cumsum(pca84.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


