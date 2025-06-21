import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)
X_b = np.c_[np.ones((100,1)), X]

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)

plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(theta), "r-")
plt.show()