import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import triang

zeros = np.zeros(60)
t1 = zeros.copy()
t1[0:10] = triang(10)
t2 = zeros.copy()
t2[5:20] = triang(15)
t3 = zeros.copy()
t3[15:35] = triang(20)
t4 = zeros.copy()
t4[25:50] = triang(25)

plt.plot(np.arange(60),zeros)
plt.plot(t1)
plt.plot(t2)
plt.plot(t3)
plt.plot(t4)
plt.show()
