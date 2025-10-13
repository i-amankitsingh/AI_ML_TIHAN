import pandas as pd
import matplotlib.pyplot as plt

serias = pd.Series([1,3,4])
# print(serias)

df = pd.DataFrame({
    "SN": [1,2,3],
    "Name": ['Ankit', 'Sachin', 'Anuj']
})

# print(df)

df = pd.read_csv("./Pandas/titanic_data.csv")
# print(data.head())
# print(data.tail())
# print(data.describe())
# print(data.info())
# print(data.shape)
# print(df.columns) Returns col label and data type
# print(df.rename(columns={"alive": "isAlive"}, inplace=True)) rename the column name or label

# print(df['alive'].describe())
# print(df.describe(include="all"))

# print(df.isnull())
# print(df['alive'].isnull())
# print(df.isnull().sum())

# print(df[df['age'].isin([25, 30])].shape[0])
# print(df['age'].values) Returns all value of that column
# print(df['age'].count())
# print(df['age'].value_counts()) Count specific value frequency in the column
# print(df['age'].unique()) Return all value ones like DISTINCT like SQL
# print(df['age'].nunique()) Count of DISTINC value
# print(pd.crosstab(df['sex'], df['age'])) compare sex with age like specifc sex exist in the specific age

# print(df[~df['sex'].isin(['male'])]) ~ for the not like !
# print(df.loc(['age'] > 50, ['age', 'sex'])) return age and sex column with 50+ age
# print(df.loc[0: 10, ['age', 'sex']]) return first 10 rows but label based
# print(df.iloc[0: 10, [1,2,13]]) return first 10 rows but index based 
# print(pd.cut(df['age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])) # To add new column
# print(df.groupby('sex')['age'].mean())
# print(df.groupby(['sex', 'class'])['age'].mean())
# print(df.groupby('class').agg({
#     'fare': ['mean', 'median', 'size'],
#     'age': ['mean']
# }))

# print(df.groupby('class')['fare'].transform('mean'))
# print(df.var(numeric_only=True))
# print(df.std(numeric_only=True))
# print(df.mode().iloc[0])
# print(df.nlargest(5, 'fare'))
# print(df.sort_values('fare').head())
# print(df.sort_index())

# df.plot(x='age', y='fare', kind='bar')
# df.plot(x='age', y='fare', kind='line')
# df.plot(x='age', y='fare', kind='scatter')
# df.groupby('sex')['sex'].count().plot(kind='pie', autopct='%1.2f%%')
# df.plot(x='sex', y='fare', kind='area', alpha=0.4)
# plt.show()


import numpy as np
print(np.empty(5))