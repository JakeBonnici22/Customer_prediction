import pandas as pd
import pandas_profiling


df = pd.read_csv('data/train_set.csv')
df_test = pd.read_csv('data/test_set.csv')


countries = pd.read_csv('data/customer_country.csv')
columns_list = df.columns.tolist()
# print(countries.columns)
# print(columns_list)

# Create a dictionary mapping "mk_CurrentCustomer" to "country"
customer_country_map = dict(zip(countries['mk_CurrentCustomer'], countries['country']))
df.insert(df.columns.get_loc("mk_CurrentCustomer") + 1, "country", df['mk_CurrentCustomer'].map(customer_country_map))
df.dropna(subset=['country'], inplace=True)
# df.to_csv('merged_countries_train.csv', index=False)

df_train = pd.read_csv('data/merged_countries_train.csv')
variable_groups = [
    'days_g', 'ro_g', 'to_g', 'gw_g', 'mar_g', 'GOC_ro_g', 'GOC_to_g',
    'GOC_dist_gm_g', 'SB_ro_g', 'SB_to_g', 'with_cnl_g', 'with_cnt_g',
    'with_sum_g', 'succ_dep_g', 'unsucc_dep_g', 'unsucc_dep_cnt_g',
    'succ_dep_cnt_g', 'pm_sum_g', 'pm_avg_g', 'ini_bon_g', 'ini_bon_cnt_g',
    'bon_wrt_succdep_g', 'turnover_last_', 'to_l', 'dwcanc_last_', 'dwcount_last_',
    'w_canc_count_ratio_l'
]
not_included = [col for col in df_train.columns if not any(group in col for group in variable_groups)]
not_included.remove('SE_total')
not_included.remove('GI_total')
not_included.remove('SE_GI_wrt_days_70days')

not_included_test = [col for col in df_test.columns if not any(group in col for group in variable_groups)]
not_included_test.remove('SE_total')
not_included_test.remove('GI_total')
not_included_test.remove('SE_GI_wrt_days_70days')

for group in variable_groups:
    group_cols = [col for col in df_train.columns if col.startswith(group)]
    df_train[f'{group}_mean'] = df_train[group_cols].mean(axis=1)

for group in variable_groups:
    group_cols = [col for col in df_test.columns if col.startswith(group)]
    df_test[f'{group}_mean'] = df_test[group_cols].mean(axis=1)

train_fe = df_train[not_included + [f'{group}_mean' for group in variable_groups]].copy()
test_fe = df_test[not_included_test + [f'{group}_mean' for group in variable_groups]].copy()


# train_fe.to_csv('data/train_fe.csv', index=False)
# test_fe.to_csv('data/test_fe.csv', index=False)


profile = pandas_profiling.ProfileReport(train_fe)
profile.to_file("output.html")


