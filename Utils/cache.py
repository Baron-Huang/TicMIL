import torch
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


p = np.array([0.34, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999])
acc_swint = np.array([88.87, 88.87, 88.47, 87.26, 86.85, 86.85, 86.85, 86.45, 84.62, 82.59, 81.79, 78.36, 75.94, 74.33])
acc_mil_swint = np.array([96.6, 96.6, 96, 95.8, 95.2, 95.2, 95, 94.6, 94.07, 91.07, 87.80, 84.53, 78.33, 69.73])

diff_swint = 0
diff_mil = 0
for i in range(acc_mil_swint.shape[0] - 1):
    diff_swint += acc_swint[i] - acc_swint[i+1]
    diff_mil += acc_mil_swint[i] - acc_mil_swint[i+1]

diff_mil = diff_mil / acc_mil_swint.shape[0]
diff_swint = diff_swint / acc_mil_swint.shape[0]

plt.figure()
plt.plot(acc_swint, '*-')
plt.plot(acc_mil_swint, '.-')
plt.grid()

p_acc_swint = np.mean(p * acc_swint) / diff_swint
p_acc_mil_swint = np.mean(p * acc_mil_swint) / diff_mil

print(p_acc_mil_swint)
print(p_acc_swint)

plt.show()
