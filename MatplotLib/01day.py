from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv('cardata.csv')

# plt.plot(df['Year'], df['Selling_Price'])
# plt.grid(True)
# plt.show()


# months = np.array(['January', 'February', 'March', 'April', 'May', 'June'])
# current_values = np.array([92483, 90384, 93947, 88310, 105385, 112478])

# plt.plot(months, current_values, ls='dotted', c='green', marker="*")
# plt.yticks([70000, 80000, 90000, 100000, 110000, 120000])
# plt.title('Investment Movement')
# plt.xlabel('Months')
# plt.ylabel('Amount')
# plt.grid()
# plt.show()

df = pd.read_csv('cardata.csv')
# df.plot(kind='bar')
print(df.describe())