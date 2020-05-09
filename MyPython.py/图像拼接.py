import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pywt
import cv2
from PIL import Image


def get_var(img):
    x, y = img.shape
    img_var = np.zeros([x, y])
    for i in range(x):
        for j in range(y):
            L = max(0, i - 5)
            R = min(x, i + 5)
            U = max(0, j - 5)
            D = min(y, j + 5)
            mean, var = cv2.meanStdDev(img[L:R, U:D])
            img_var[i, j] = var
    return img_var


def img_var(img1, img2):
    mean1, var1 = cv2.meanStdDev(img1)
    mean2, var2 = cv2.meanStdDev(img2)
    weight1 = var1 / (var1 + var2)
    weight2 = var2 / (var1 + var2)
    return weight1, weight2


def solve_low(res1, res2):
    w1, w2 = img_var(res1[0], res2[0])
    Arr = np.zeros(res2[0].shape)
    x, y = res1[0].shape
    for i in range(x):
        for j in range(y):
            Arr[i, j] = w1 * res1[0][i, j] + w2 * res2[0][i, j]
    return Arr


def solve_high(res1, res2, k):
    Arr = []
    for array1, array2 in zip(res1[k], res2[k]):
        tmp_x, tmp_y = array1.shape
        highFreq = np.zeros((tmp_x, tmp_y))
        var1 = get_var(array1)
        var2 = get_var(array2)
        for i in range(tmp_x):
            for j in range(tmp_y):
                highFreq[i, j] = array1[i, j] if var1[i, j] > var2[i, j] else array2[i, j]
        Arr.append(highFreq)
    return tuple(Arr)


def get_wave(img1, img2):
    res1 = pywt.wavedec2(img1, 'haar', level=4)
    res2 = pywt.wavedec2(img2, 'haar', level=4)
    cur_wave = []
    for k in range(len(res1)):
        # 处理低频分量
        if k == 0:
            cur_wave.append(solve_low(res1, res2))
        else:
            cur_wave.append(solve_high(res1, res2, k))
    return pywt.waverec2(cur_wave, 'haar')


if __name__ == '__main__':
    path1 = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/homework/1.jpg'
    path2 = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/homework/2.jpg'

    # 加载图像
    img1 = np.array(Image.open(path1).convert("L"))
    img2 = np.array(Image.open(path2).convert("L"))
    img3 = get_wave(img1, img2)

    # 保存图像
    image.imsave('C:/Users/boliz/Desktop/img1.png', img1, plt.gray())
    image.imsave('C:/Users/boliz/Desktop/img2.png', img2)
    image.imsave('C:/Users/boliz/Desktop/img3.png', img3)

    # 展示图像
    # fig, ax = plt.subplots(1, 3, figsize=(60, 84))
    # ax[0].imshow(img1, plt.gray())
    # ax[0].set(title="src1")
    # ax[1].imshow(img2, plt.gray())
    # ax[1].set(title="src2")
    # ax[2].imshow(img3, plt.gray())
    # ax[2].set(title="ans")
    # plt.show()
