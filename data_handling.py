import pandas as pd
import pandas_profiling


df = pd.read_csv('data/train_set.csv')
countries = pd.read_csv('data/customer_country.csv')
columns_list = df.columns.tolist()
# print(countries.columns)
# print(columns_list)

# Create a dictionary mapping "mk_CurrentCustomer" to "country"
customer_country_map = dict(zip(countries['mk_CurrentCustomer'], countries['country']))
df.insert(df.columns.get_loc("mk_CurrentCustomer") + 1, "country", df['mk_CurrentCustomer'].map(customer_country_map))
df.dropna(subset=['country'], inplace=True)
df.to_csv('merged_countries_train.csv', index=False)


profile = pandas_profiling.ProfileReport(df_train)
profile.to_file("output.html")