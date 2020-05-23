import nn
import numpy as np
import types
import cv2
import time
import matplotlib.pyplot as plt

        
if __name__ == "__main__":
    print('shit change git test')
    img = cv2.imread('data/cats.jpg')
    img = img[:900, 250:1150, :]
    print(img.shape)
    img = img / 255.0
    plt.imshow(img)
    plt.show()
    x = np.random.rand(32, 32, 3)
    conv1 = nn.Conv2d(3, 64, k_size=3, stride=1, padding=0)
    relu = nn.Relu()
    t1 = time.time()
    out = conv1(np.array([img]))
    t2 = time.time()
    print('cost time: {}'.format(t2-t1))
    out = relu(out)
    print('relu cost: ', time.time()-t2)
    for i in range(64):
        im = out[0, :, :, i].reshape(out.shape[1], out.shape[2], 1)
        print(im.shape)
        cv2.imshow('img', im)
        cv2.waitKey(500)

