# PyTorch test
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print("PyTorch tensor:", x)

# TensorFlow test
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0])
print("TensorFlow tensor:", a)

# Scikit-Learn test
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression()
model.fit(X, y)
print("Scikit-Learn prediction for 5:", model.predict([[5]]))
