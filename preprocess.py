import pandas as pd
import numpy as np

df = pd.read_csv('data/train_set.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

columns_list = df.columns.tolist()
print(columns_list)