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

df_train = pd.read_csv('data/train_fe.csv')
df_test = pd.read_csv('data/test_fe.csv')

# PCA to reduce dimensionality of the features to 95% variance and reducing computational resources/time needed.
n_components = df_train.shape[1] - 4
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df_train.drop(columns=['country', 'target', 'mk_CurrentCustomer', 'ScoreDate']))
X_pca_test = pca.transform(df_test.drop(columns=['ScoreDate', 'mk_CurrentCustomer']))
df_pca_test = pd.DataFrame(data=X_pca_test, columns=[f'Component_{i+1}' for i in range(n_components)], index=df_test.index)
df_pca_test['mk_CurrentCustomer'] = df_test['mk_CurrentCustomer']


explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

feature_names = df_train.drop(columns=['country', 'target', 'mk_CurrentCustomer', 'ScoreDate']).columns

print("Feature names chosen by PCA:")
for i in range(n_components):
    component_index = np.argmax(np.abs(pca.components_[i]))
    feature_name = feature_names[component_index]
    print(f"Component {i+1}: {feature_name}")


# Plot explained variance ratio and cumulative variance ratio
plt.figure()
plt.plot(range(1, n_components + 1), explained_variance_ratio, 'b-o', label='Explained Variance Ratio')
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'r-o', label='Cumulative Variance Ratio')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
index_threshold = np.argmax(cumulative_variance_ratio >= 0.99) + 1
plt.axvline(x=index_threshold, color='k', linestyle=':', label='Selected Components')
plt.text(index_threshold + 1, cumulative_variance_ratio[index_threshold - 1], f'{index_threshold}', va='center')
plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio')
plt.legend()
plt.savefig('figures/pca_variance_ratio.png')
plt.show()

chosen_feature_names = feature_names[:index_threshold]

X_train, X_val, y_train, y_val = train_test_split(X_pca,
                                                  df_train['target'],
                                                  test_size=0.2, random_state=42,
                                                  stratify=df_train['target'])


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print()
print("X_test shape:", df_pca_test.shape)
print("Preprocessing done")

