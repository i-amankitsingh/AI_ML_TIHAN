import matplotlib as plt
import pandas as pd

df = pd.read_csv('cardata.csv')

plt.plot(df['Year'], df['Selling_Price'])
plt.show()