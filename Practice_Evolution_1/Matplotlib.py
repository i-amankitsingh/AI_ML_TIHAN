import matplotlib.pyplot as plt
import numpy as np

x = np.array(['Ankit', 'Sachin', 'Anuj', 'Manish'])
y = np.array([422, 409, 396, 394])

# plt.plot(x, y, color="green", marker="*")
# plt.plot(x,y, 'o:r')
# plt.plot(x,y, marker='o', ms=20, mec='r', mfc='g')
plt.plot(x,y, linestyle=':')
plt.show()