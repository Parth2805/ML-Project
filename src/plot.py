import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(4, 1, 100)
plt.interactive(True)
plt.plot(x, x)
print("hello")
plt.show(block=True)