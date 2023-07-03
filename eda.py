import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('data/merged_countries_train.csv')
fe_train = pd.read_csv('data/test_fe.csv')

print(fe_train.columns.tolist())
value_counts = df_train['target'].value_counts()
percentage_1s = (value_counts[1] / len(df_train['target'])) * 100
percentage_0s = (value_counts[0] / len(df_train['target'])) * 100
print("Percentage of 1s: {:.2f}%".format(percentage_1s))
print("Percentage of 0s: {:.2f}%".format(percentage_0s))
print(df_train.columns.tolist())
print(df_train.isna().sum())


columns_to_plot = ['GOC_to_g_mean', 'SB_to_g_mean', 'ro_g_mean']
fe_train_columns = ['mk_CurrentCustomer'] + columns_to_plot
merged_df = pd.merge(fe_train[fe_train_columns], df_train[['mk_CurrentCustomer', 'target']], on='mk_CurrentCustomer')
data_melted = merged_df.melt(id_vars=['target', 'mk_CurrentCustomer'], var_name='Column', value_name='Value')
column_order = ['GOC_to_g_mean', 'SB_to_g_mean', 'ro_g_mean']
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Column', y='Value', hue='target', data=data_melted, order=column_order)
# plt.xlabel('Column', fontsize=14)
plt.ylabel('Mean Value')
plt.title('Mean Values of GOC, SB and game rounds grouped by Target', fontsize=16)
legends = ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig('figures/Mean Value by Column and Target.png')
plt.show()


columns_to_plot = ['turnover_last_1day', 'turnover_last_2days', 'turnover_last_3days',
                   'turnover_last_5days', 'turnover_last_10days', 'turnover_last_20days', 'turnover_last_70days']
data_to_plot = df_train[columns_to_plot + ['target']]
data_melted = data_to_plot.melt(id_vars='target', var_name='Turnover Period', value_name='Turnover')
mean_turnover = data_melted.groupby(['Turnover Period', 'target'])['Turnover'].mean().reset_index()
mean_turnover['Turnover Period'] = pd.Categorical(mean_turnover['Turnover Period'], categories=columns_to_plot, ordered=True)
plt.figure(figsize=(12, 10))
mean_turnover = mean_turnover.sort_values('Turnover Period')
ax = sns.barplot(x='Turnover Period', y='Turnover', hue='target', data=mean_turnover)
plt.xlabel('Turnover Period', fontsize=16)
plt.ylabel('Mean Turnover', fontsize=16)
plt.title('Mean Turnover by Turnover Period and Target', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
legends = ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig('figures/Mean Turnover by Turnover Period and Target.png')
plt.show()


class_counts = df_train['target'].value_counts()
total_count = class_counts.sum()
percentages = (class_counts / total_count) * 100
ax = class_counts.plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Class Distribution')
for i, count in enumerate(class_counts):
    plt.text(i, count + 0.5, f'{percentages[i]:.2f}%', ha='center')
plt.savefig('figures/Class Distribution.png')
plt.show()


# Distribution of 0 and 1 values by Country
grouped_data = df_train.groupby(['country', 'target']).size().unstack(fill_value=0)
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(grouped_data))
ax.bar(index, grouped_data[0], bar_width, label='0')
ax.bar(index, grouped_data[1], bar_width, bottom=grouped_data[0], label='1')
ax.set_xticks(index)
ax.set_xticklabels(grouped_data.index)
ax.set_ylabel('Count')
ax.set_title('Distribution of 0 and 1 values by Country')
ax.legend()
plt.savefig('figures/Distribution of 0 and 1 values by Country.png')
plt.show()


# Distribution of 0 and 1 values by Country
activity_columns = df_train.columns[df_train.columns.str.startswith('days_g')].tolist()
target_column = 'target'
selected_columns = activity_columns + [target_column]
df_selected = df_train[selected_columns]
grouped_data = df_selected.groupby(target_column).sum()
grouped_data = grouped_data.T
fig, ax = plt.subplots()
week_labels = ['Week ' + str(i) for i in range(1, 11)]
ax.set_xticks(range(len(grouped_data.index)))
ax.set_xticklabels(week_labels)
ax.set_ylabel('Activity Count')
ax.set_title('Activity Distribution by Target')
x = range(len(grouped_data.index))
width = 0.35
for i, column in enumerate(grouped_data.columns):
    ax.bar([val + i * width for val in x], grouped_data[column], width, label=column)
ax.legend()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.savefig('figures/Activity Distribution by Target.png')
plt.show()


# Mean and Median of the number of days of activity during the week
selected_columns = ['days_g{}'.format(i) for i in  range(0,11,1)]
df_gx = pd.DataFrame(df_train, columns=selected_columns)
mean_values = df_gx.mean()
median_values = df_gx.median()
mean_values.plot.line(marker='o', label='Mean')
median_values.plot.line(marker='s', label='Median')
plt.title('Mean and Median of the number of days of activity during the week')
plt.xlabel('Weeks')
plt.ylabel('Days')
plt.legend()
plt.savefig('figures/mean_median_activities.png')
plt.show()


# turnover,game winnings mean and median
turnover = ['to_g{}'.format(i) for i in  range(1,11,1)]
game_wins = ['gw_g{}'.format(i) for i in  range(1,11,1)]
mar_g = ['mar_g{}'.format(i) for i in  range(1,11,1)]
df_gt = pd.DataFrame(df_train, columns=turnover)
df_gw = pd.DataFrame(df_train, columns=game_wins)
df_marg = pd.DataFrame(df_train, columns=mar_g)
mean_gt = df_gt.mean()
median_gt = df_gt.median()
mean_gw = df_gw.mean()
median_gw = df_gw.median()
mean_marg = df_marg.mean()
median_marg = df_marg.median()


# Plotting Total Turnover
mean_gt.plot.line(marker='o', linestyle='-', color='blue', label='Total Turnover Mean (Euro)')
median_gt.plot.line(marker='o', linestyle='--', color='blue', label='Total Turnover Median (Euro)')

# Plotting Game Winnings
mean_gw.plot.line(marker='o', linestyle='-', color='green', label='Game Winnings Mean (Euro)')
# median_gw.plot.line(marker='o', linestyle='--', color='green', label='Game Winnings Median (Euro)')

# Plotting Total Margin
mean_marg.plot.line(marker='o', linestyle='-', color='red', label='Total Margin Mean (Euro)')
# median_marg.plot.line(marker='o', linestyle='--', color='red', label='Total Margin Median (Euro)')

plt.title('Mean and Median of the total turnover/game_winnings/margin amount in Euro during the weeks')
plt.xlabel('Weeks')
plt.ylabel('Turnover')
plt.legend()
plt.ylim(bottom=-500)
plt.savefig('figures/mean_median_to_t_gw_mar.png')
plt.show()


# Successful and Cancelled Withdraws
mean_cnl = df_train.filter(like='with_cnl').mean()
mean_cnt = df_train.filter(like='with_cnt').mean()

weeks = np.arange(1, 11)
bar_width = 0.2

plt.bar(weeks - bar_width, mean_cnl, width=bar_width, color='#1992D4', label='Cancelled Withdraws (Mean)')
plt.bar(weeks, mean_cnt, width=bar_width, color='#085B8E', label='Successful Withdraws (Mean)')
plt.xlabel('Weeks')
plt.ylabel('Number of Withdraws')
plt.title('Withdraws by Week')
plt.xticks(weeks, ['Week ' + str(w) for w in weeks])
plt.legend()
plt.savefig('figures/withdraws.png')
plt.show()


# Successful and Cancelled Deposits
mean_succ_dep = ['succ_dep_cnt_g{}'.format(i) for i in range(1,11,1)]
mean_succ_dep = df_train[mean_succ_dep].mean()

mean_unsucc_dep = df_train.filter(like='unsucc_dep_cnt_g').mean()
print(mean_succ_dep)
weeks = np.arange(1, 11)
bar_width = 0.2

plt.bar(weeks - bar_width, mean_succ_dep, width=bar_width, color='#1992D4', label='Successful Deposits (Mean)')
plt.bar(weeks, mean_unsucc_dep, width=bar_width, color='#085B8E', label=' Unsuccessful Deposits (Mean)')
plt.xlabel('Weeks')
plt.ylabel('Number of Deposits')
plt.title('Withdraws by Week')
plt.xticks(weeks, ['Week ' + str(w) for w in weeks], rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig('figures/deposits.png')
plt.show()


df_train['GI_total'].fillna(0, inplace=True)
print(df_train['GI_total'].unique())
count_non_zero = (df_train['GI_total'] != 0).sum()
print(count_non_zero)

df_train['SE_total'].fillna(0, inplace=True)
print(df_train['SE_total'].unique())
count_non_zero = (df_train['SE_total'] != 0).sum()
print(count_non_zero)


# average total number of payment methods for target 0 and 1 group during each week.
target_0 = df_train[df_train['target'] == 0]
avg_pm_sum_g_0 = target_0[['pm_sum_g1', 'pm_sum_g2', 'pm_sum_g3', 'pm_sum_g4', 'pm_sum_g5', 'pm_sum_g6', 'pm_sum_g7',
                           'pm_sum_g8', 'pm_sum_g9', 'pm_sum_g10']].mean()
target_1 = df_train[df_train['target'] == 1]
avg_pm_sum_g_1 = target_1[['pm_sum_g1', 'pm_sum_g2', 'pm_sum_g3', 'pm_sum_g4', 'pm_sum_g5', 'pm_sum_g6', 'pm_sum_g7',
                           'pm_sum_g8', 'pm_sum_g9', 'pm_sum_g10']].mean()
x = np.arange(10)
width = 0.40
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, avg_pm_sum_g_0, width, label='Target 0')
rects2 = ax.bar(x + width/2, avg_pm_sum_g_1, width, label='Target 1')
ax.set_ylabel('Average of the total Payment Methods')
ax.set_title('Average of the total Payment Methods by week')
ax.set_xticks(x)
ax.set_xticklabels(['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8', 'Week 9',
                    'Week 10'], rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig('figures/total Payment Methods by week.png')
plt.show()


# correlation with target
fe_train_merged = pd.merge(fe_train, df_train[['mk_CurrentCustomer', 'target']], on='mk_CurrentCustomer', how='left')
correlation_matrix = fe_train_merged.corr()
target_correlation = correlation_matrix['target'].drop('target')
plt.figure(figsize=(8, 10))
sns.heatmap(pd.DataFrame(target_correlation), annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Correlation with Target')
plt.savefig('figures/correlation_with_target.png')
plt.show()

