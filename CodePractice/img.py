import pandas as pd
from sklearn.linear_model import LinearRegression

# Small House Price Data
data = {
    'Area': [1000, 1500, 2000, 2500, 1800, 1200, 2200, 1600],
    'Bedrooms': [2, 3, 3, 4, 3, 2, 4, 3],
    'Age': [10, 8, 5, 3, 12, 15, 7, 9],
    'Price': [50, 65, 80, 100, 75, 55, 90, 70]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Model training
model = LinearRegression()
model.fit(X, y)

# R-squared
r_squared = model.score(X, y)

# Adjusted R-squared Formula
n = len(y)   # number of data points
k = X.shape[1]   # number of predictors

adjusted_r2 = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r2)
