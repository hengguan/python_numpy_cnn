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
    # plt.imshow(img)
    # plt.show()
    # x = np.random.rand(32, 32, 3)
    out_img = None
    for i in range(3):
        conv = nn.Conv2d(1, 1, k_size=3, stride=1, padding=1)
        in_img = img[:, :, i]
        out = conv(np.array([in_img[:, :, np.newaxis]]))
        out_img = np.concatenate((out_img, out[0]), axis=-1) \
            if out_img is not None else out[0]
        
    cv2.imshow('conv', out_img)
    cv2.waitKey(0)
    # relu = nn.Relu()
    # pool = nn.MaxPooling(k_size=2, stride=2)
    # t1 = time.time()
    # out = conv1(np.array([img]))
    # t2 = time.time()
    # out = relu(out)
    # t3 = time.time()
    # out = pool(out)
    # t4 = time.time()
    # print(out.shape)
    # print('cost time: {}'.format(t2-t1))
    # print('relu cost: ', t3-t2)
    # print('maxpooling cost: ', t4-t3)
    for i in range(64):
        im = out[0, :, :, i]  # .reshape(out.shape[1], out.shape[2], 1)
        print(im.shape)
        cv2.imshow('img', im)
        cv2.waitKey(500)

