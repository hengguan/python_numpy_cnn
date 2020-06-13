import numpy as np


a = np.array(range(150)).reshape(2, 5, 5, 3)
# print(a)
m = np.mean(a, axis=-1)
print(m.shape)
b = np.arange(9).reshape(3, 3)
print(b*b)
print(b**2)

# target = np.array([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0, 1]])

# exp_a = np.exp(a)
# sum_exp = np.sum(exp_a, axis=0, keepdims=True)
# print(exp_a)
# print(sum_exp)
# print(exp_a/sum_exp)

# print(-target*np.log(exp_a/sum_exp))
