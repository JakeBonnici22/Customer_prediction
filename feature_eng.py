import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('data/merged_countries_train.csv')
df_test = pd.read_csv('data/test_set.csv')

n_components = 10
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df_train.drop(columns=['country', 'target', 'mk_CurrentCustomer', 'ScoreDate']))
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.figure()
plt.plot(range(1, n_components + 1), explained_variance_ratio, 'b-o', label='Explained Variance Ratio')
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'r-o', label='Cumulative Variance Ratio')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
index_threshold = np.argmax(cumulative_variance_ratio >= 0.95) + 1
plt.axvline(x=index_threshold, color='k', linestyle=':', label='Selected Components')
plt.text(index_threshold + 1, cumulative_variance_ratio[index_threshold - 1], f'{index_threshold}', va='center')
plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio')
plt.legend()
plt.show()


columns_to_drop_train = ['SE_total', 'GI_total', 'SE_GI_total', 'SE_GI_total_70days', 'country', 'target',
                         'mk_CurrentCustomer', 'ScoreDate']
columns_to_drop_test = ['SE_total', 'GI_total', 'SE_GI_total', 'SE_GI_total_70days', 'mk_CurrentCustomer', 'ScoreDate']

X_test = df_test.drop(columns=columns_to_drop_test)
X_train, X_val, y_train, y_val = train_test_split(X_pca,
                                                  df_train['target'],
                                                  test_size=0.2, random_state=42,
                                                  stratify=df_train['target'])


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

X_test = df_test.drop(columns=columns_to_drop_test)


print("X_test shape:", X_test.shape)




print()
print("Preprocessing done")

