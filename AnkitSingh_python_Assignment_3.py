import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
from sklearn.pipeline import Pipeline

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense

import gradio as gr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston


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


# 5
df = pd.read_csv("./MachineLearning/uav_raw.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["alt_fused_m"] = 0.7 * df["alt_baro_m"] + 0.3 * df["alt_gps_m"]

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["alt_baro_m"], label="Baro Altitude (m)")
plt.plot(df["timestamp"], df["alt_gps_m"], label="GPS Altitude (m)")
plt.plot(df["timestamp"], df["alt_fused_m"], label="Fused Altitude (m)")

plt.title("Altitude Comparison: Barometer vs GPS vs Fused")
plt.xlabel("Time")
plt.ylabel("Altitude (m)")
plt.legend()
plt.tight_layout()
plt.show()


#6

df = pd.read_csv("./MachineLearning/uav_raw.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["battery_pct"], marker="o", linestyle="-", label="Battery %")

plt.title("Battery Percentage Over Time")
plt.xlabel("Time")
plt.ylabel("Battery (%)")
plt.ylim(0, 100) 
plt.legend()
plt.tight_layout()
plt.show()


# 7
df = pd.read_csv("./MachineLearning/uav_raw.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.drop_duplicates()

df = df.sort_values(by="timestamp").reset_index(drop=True)

print(df.head())  


# 8

df = pd.read_csv("./MachineLearning/uav_raw.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.drop_duplicates().sort_values(by="timestamp").reset_index(drop=True)


df["battery_pct"] = pd.to_numeric(df["battery_pct"], errors="coerce") 
df["battery_pct"] = df["battery_pct"].clip(lower=0, upper=100)

print(df.head()) 


# ✅ D) AI-Powered App Development

# 1

tips = sns.load_dataset("tips")

df = pd.get_dummies(tips, drop_first=True)

X = df.drop("tip", axis=1)
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)


# 2

tips = sns.load_dataset("tips")


df = pd.get_dummies(tips, drop_first=True)


X = df.drop("tip", axis=1)
y = df["tip"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


def predict_tip(total_bill, size, sex, smoker, day, time):
   
    input_dict = {
        "total_bill": [total_bill],
        "size": [size],
        "sex_Male": [1 if sex == "Male" else 0],
        "smoker_Yes": [1 if smoker == "Yes" else 0],
        "day_Sat": [1 if day == "Sat" else 0],
        "day_Sun": [1 if day == "Sun" else 0],
        "day_Thur": [1 if day == "Thur" else 0],
        "time_Dinner": [1 if time == "Dinner" else 0],
    }
    
    input_df = pd.DataFrame(input_dict)
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)


interface = gr.Interface(
    fn=predict_tip,
    inputs=[
        gr.Number(label="Total Bill ($)"),
        gr.Number(label="Party Size"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Radio(["Yes", "No"], label="Smoker"),
        gr.Radio(["Thur", "Fri", "Sat", "Sun"], label="Day"),
        gr.Radio(["Lunch", "Dinner"], label="Time"),
    ],
    outputs=gr.Number(label="Predicted Tip ($)"),
    title="Tip Prediction App 💡",
    description="Enter details to predict the tip amount using Linear Regression."
)

interface.launch()


# 3


tips = sns.load_dataset("tips")


X = tips.drop("tip", axis=1)
y = tips["tip"]


categorical_cols = ["sex", "smoker", "day", "time"]
numeric_cols = ["total_bill", "size"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model.fit(X_train, y_train)


def predict_tip(total_bill, size, sex, smoker, day, time):
    input_df = pd.DataFrame([{
        "total_bill": total_bill,
        "size": size,
        "sex": sex,
        "smoker": smoker,
        "day": day,
        "time": time,
    }])
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)


interface = gr.Interface(
    fn=predict_tip,
    inputs=[
        gr.Number(label="Total Bill ($)"),
        gr.Number(label="Party Size"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Radio(["Yes", "No"], label="Smoker"),
        gr.Radio(["Thur", "Fri", "Sat", "Sun"], label="Day"),
        gr.Radio(["Lunch", "Dinner"], label="Time"),
    ],
    outputs=gr.Number(label="Predicted Tip"),
    title="Tip Prediction App",
    description="Enter details to predict the tip amount using Linear Regression."
)

interface.launch()


#4

tips = sns.load_dataset("tips")


X = tips.drop("tip", axis=1)
y = tips["tip"]


categorical_cols = ["sex", "smoker", "day", "time"]
numeric_cols = ["total_bill", "size"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


# 5

ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cols = ohe.get_feature_names_out(categorical_cols)
all_features = list(encoded_cols) + numeric_cols


coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": all_features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)


plt.figure(figsize=(10, 5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="skyblue")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# 6

tips = sns.load_dataset("tips")


X = tips.drop("tip", axis=1)
y = tips["tip"]

categorical_cols = ["sex", "smoker", "day", "time"]
numeric_cols = ["total_bill", "size"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)


mlp_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MLPRegressor(hidden_layer_sizes=(64, 32), 
                               activation="relu",
                               solver="adam",
                               max_iter=1000,
                               random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


mlp_model.fit(X_train, y_train)


y_pred = mlp_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MLP Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# 7

iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(
    hidden_layer_sizes=(50, 30),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"MLP Accuracy on Test Set: {accuracy:.2f}")
'''

# 8

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])


model.fit(X_train, y_train)

def predict_price(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT):
    input_data = pd.DataFrame([[
        CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
    ]], columns=boston.feature_names)
    
    prediction = model.predict(input_data)[0]
    return f"Predicted House Price: ${prediction*1000:.2f}"


interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="CRIM (Per capita crime rate)"),
        gr.Number(label="ZN (Residential land zoned %)"),
        gr.Number(label="INDUS (Non-retail business acres %)"),
        gr.Number(label="CHAS (Charles River dummy variable: 0 or 1)"),
        gr.Number(label="NOX (Nitric oxide concentration)"),
        gr.Number(label="RM (Average rooms per dwelling)"),
        gr.Number(label="AGE (Proportion of old units)"),
        gr.Number(label="DIS (Distance to employment centers)"),
        gr.Number(label="RAD (Accessibility to highways)"),
        gr.Number(label="TAX (Property tax rate)"),
        gr.Number(label="PTRATIO (Pupil-teacher ratio)"),
        gr.Number(label="B (1000*(Bk - 0.63)^2, where Bk is proportion of blacks)"),
        gr.Number(label="LSTAT (% lower status population)"),
    ],
    outputs="text",
    title="House Price Prediction App",
    description="Enter housing details to predict the median value of a home in $1000s (Boston Housing Dataset)."
)

interface.launch()