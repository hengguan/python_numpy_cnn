import numpy as np

if __name__ == "__main__":
    a = np.arange(6).reshape(2, 1, 3)
    b = np.arange(108).reshape(36, 3)
    res = a*b
    print(res.shape)
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
