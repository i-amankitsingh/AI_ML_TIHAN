import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import time
from pyspark.sql import SparkSession

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# import torch

# import tensorflow as tf

import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense

'''
# ✅ A ML Foundations & Data Preprocessing
# 1
arr = np.arange(1, 11)
arr_sq = arr ** 2
print(arr_sq)

# 2
x = np.linspace(-10, 10, 100)
y_linear = 2** x + 1

y_non_linear = x**2

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)   # 1 row, 2 cols, 1st plot
plt.plot(x, y_linear, 'b', label="y = 2x + 1")
plt.title("Linear Relationship")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)  
plt.plot(x, y_non_linear, 'r', label="y = x²")
plt.title("Non-linear Relationship")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#3
tips = sns.load_dataset("tips")
sns.boxenplot(x=tips["total_bill"])
plt.show()


#4
tips = sns.load_dataset("tips")
mean = tips["total_bill"].mean()
median = tips["total_bill"].median()
print("mean:- ", mean)
print("median:- ", median)


# 5
df = pd.read_csv("sale_data.csv")
df_encoded = pd.get_dummies(df, columns=["Category"])
print(df_encoded)




# 6
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([3, 4, 2, 5, 7, 8, 9, 10, 12, 13])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Coefficients (slope):", model.coef_)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)


# 7
tips = sns.load_dataset("tips")
x = tips[["total_bill"]]
y = tips["tip"]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

rmse_with_outliers = sqrt(mean_squared_error(y, y_pred))

Q1 = tips["total_bill"].quantile(0.25)
Q3 = tips["total_bill"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 - 1.5 * IQR

tips_no_outliers = tips[(tips["total_bill"] >= lower_bound) & (tips["total_bill"] <= upper_bound)]
x_no = tips_no_outliers[["total_bill"]]
y_no = tips_no_outliers["tip"]

mode_no = LinearRegression()
mode_no.fit(x_no, y_no)
y_pred_no = mode_no.predict(x_no)

rmse_without_outliers = sqrt(mean_squared_error(y_no, y_pred_no))

print("RMSE with outliers: ", rmse_with_outliers)
print("RMSE without outliers: ", rmse_without_outliers)


# 8

data = np.random.rand(10_000_000)
df = pd.DataFrame({"values": data})

start = time.time()
mean_val = df["values"].mean()
end = time.time()

print("Pandas mean:", mean_val)
print("Execution Time:", end - start, "seconds")

spark = SparkSession.builder.appName("MeanDemo").getOrCreate()

data = [(float(i),) for i in range(100_000_000)]
df = spark.createDataFrame(data, ["values"])

start = time.time()
mean_val = df.groupBy().avg("values").collect()[0][0]
end = time.time()

print("Spark mean:", mean_val)
print("Execution Time:", end - start, "seconds")


# ✅ B) ML Libraries
# 1

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


# 2

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[5.0], [7.0], [9.0], [11.0]])

w = torch.randn(1, required_grad=True)
b = torch.randn(1, required_grad=True)

lr = 0.01

for epoch in range(100):
    
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)
    
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
    w.grand.zero_()
    b.grand.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print("\nFinal Parameters: ", w.item(), b.item())


# 3

X = np.random.rand(100, 1).astype(np.float32)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1).astype(np.float32)  

W = tf.Variable(tf.random.normal([1, 1]))
b = tf.Variable(tf.zeros([1]))


lr = 0.1

for epoch in range(1000):
    with tf.GradientTape() as tape:
        
        y_pred = tf.matmul(X, W) + b
        
        loss = tf.reduce_mean(tf.square(y - y_pred))

    grads = tape.gradient(loss, [W, b])
    
    W.assign_sub(lr * grads[0])
    b.assign_sub(lr * grads[1])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss.numpy():.4f}, W={W.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")

print("\nTrained Parameters:")
print("Weight (W):", W.numpy()[0][0])
print("Bias (b):", b.numpy()[0])


# 4

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("logreg_iris.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model converted and saved as logreg_iris.onnx")


# 5

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

start = time.time()

sk_model = LogisticRegression(max_iter=1000)
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)

end = time.time()
sk_time = end - start
sk_acc = accuracy_score(y_test, sk_pred)

print("Scikit-learn Logistic Regression:")
print(f"⏱ Training Time: {sk_time:.4f} sec")
print(f"Accuracy: {sk_acc:.4f}")


start = time.time()

tf_model = Sequential([
    Dense(10, activation="relu", input_shape=(X.shape[1],)),
    Dense(3, activation="softmax")
])
tf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

tf_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0) 
tf_loss, tf_acc = tf_model.evaluate(X_test, y_test, verbose=0)

end = time.time()
tf_time = end - start

print("\nTensorFlow Neural Network:")
print(f"⏱ Training Time: {tf_time:.4f} sec")
print(f"Accuracy: {tf_acc:.4f}")


# 6

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)

B = torch.tensor([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=torch.float32)

C = torch.matmul(A, B) 

print("Tensor A:\n", A)
print("Tensor B:\n", B)
print("Result of A x B:\n", C)


# 7

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[5.0], [7.0], [9.0], [11.0]])

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


lr = 0.01


optimizer = torch.optim.SGD([w, b], lr=lr)


for epoch in range(100):
    y_pred = w * X + b

    loss = torch.mean((y_pred - y) ** 2)

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()        

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print("\nFinal Parameters:")
print(f"w = {w.item():.4f}, b = {b.item():.4f}")



# ✅ C) UAV Data Preprocessing
# 1

df = pd.read_csv("./MachineLearning/uav_raw.csv")
df['speed_ms'] = df['speed_kmh'] * [1000/3600]
print(df)


# 2
df = pd.read_csv("./MachineLearning/uav_raw.csv")
df['alt_gps_m'] = df['alt_gps_m'].interpolate(method="linear")

df['alt_gps_m'] = df['alt_gps_m'].fillna(method='bfill').fillna(method='ffill')

print(df[['timestamp', 'alt_baro_m', 'alt_gps_m']].head(10))


# 3

df = pd.read_csv("./MachineLearning/uav_raw.csv")
df['alt_gps_m']  = df['alt_gps_m'].interpolate('linear').bfill().ffill()
df['alt_baro_m'] = df['alt_baro_m'].interpolate('linear').bfill().ffill()

w_gps, w_baro = 0.7, 0.3

g = df['alt_gps_m']
b = df['alt_baro_m']
fused = np.where(g.notna() & b.notna(), w_gps*g + w_baro*b, g.fillna(b))

df['alt_fused_m'] = fused

print(df[['alt_baro_m','alt_gps_m','alt_fused_m']].head())


# 4

df = pd.read_csv("./MachineLearning/uav_raw.csv")

df["roll"] = np.arctan2(df["ay_ms2"], df["az_ms2"]) * 180/np.pi
df["pitch"] = np.arctan2(-df["ax_ms2"], np.sqrt(df["ay_ms2"] ** 2 + df["az_ms2"] ** 2)) * 180/np.pi

print(df)
'''