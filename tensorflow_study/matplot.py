import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%matplotlib inline

plt.rcParams['figure.figsize'] = (10,6)

X = np.arange(0.0,5.0,0.1)

a = 1
b = 0

Y = a*X+b

plt.plot(X,Y)
plt.ylabel('Y')
plt.xlabel('X')
plt.show()
