import pandas as pd


data = pd.read_csv(f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
data.to_csv("titanic_data.csv", index=False)

# print(data.head())
# print(data.tail())
# print(data.info())
# print(data["age"].describe())
# print(data.describe(include="all"))
# print(data.isnull())
# print(data.isnull().sum())

# print(data[['sex', 'age']].head())
# print(data[(data['age'] > 50) & (data['sex'] == 'male')])
# print(data[data['age'].isin([26, 30])])
# print(data['age'].value_counts())
# print(data['age'].unique())
# print(data['age'].nunique())
# print(pd.crosstab(data['sex'], data['age']))
# print(data[~data['sex'].isin(['male'])]) For not male in the sex
# print(data.loc[data['age'] > 50, ['age', 'sex']])
# print(data.loc[0: 10, ['age', 'fare']]) label based indexing
# print(data.iloc[0: 10, [1, 4, 7]]) number index based indexing
# print(data.head())
# print(data[(data['sex'] == 'male') & (data['class'] == 'Third') & (data['fare'] > 30)])
# print(data[(data['survived'] == 1) & (data['age'] < 10)])
# data["age_group"] = pd.cut(data['age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior']) To add new column
# data["is_old"] = data['age'] > 60 To add new col with True or False
# print(data.groupby('sex')['age'].mean())
# # print(data.groupby(['sex', 'class'])['age'].mean())
# print(data.groupby('class').agg({
#     'fare': ['max', 'min', 'mean'],
#     'age': 'median'
# }))
# data['fare_mean_by_class'] = data.groupby('class')['fare'].transform('mean') to add new column with aggrate value with same df structure
# print(data.groupby(['sex', 'class'])['survived'].mean())
# print(data.groupby(["embarked"])[['age', 'fare']].mean())
# print(data.var(numeric_only=True))
# print(data.std(numeric_only=True))
# print(data.mode().iloc[0])
# print(data.max(numeric_only=True))
# print(data[['survived', 'age']].corr())
# print(data.nlargest(5, 'fare'))
# print(data.sort_values('age').head())
# print(data[data['survived'] == 1].sort_values(by='fare', ascending=False).head())
# data['age'] = data['age'].fillna(data['age'].mean())
# print(data.rename(columns={"fare": "ticket: fare"}))
# minors_data = data[data['age'] < 18]
# minors_data.to_csv("minors.csv", index=False)