import numpy as np


a = np.array(range(9)).reshape(3, 3)
print(a)
print(np.mean(a, 0))
print(np.random.random())

target = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]])

exp_a = np.exp(a)
sum_exp = np.sum(exp_a, axis=0, keepdims=True)
print(exp_a)
print(sum_exp)
print(exp_a/sum_exp)

print(-target*np.log(exp_a/sum_exp))
