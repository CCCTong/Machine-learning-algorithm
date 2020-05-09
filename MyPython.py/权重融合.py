import cv2
import numpy as np


# 求解一张图片局部方差
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


# 求解融合时刻两张图片的权重
def get_pow(img1, img2):
    x, y = img1.shape
    var1 = get_var(img1)
    var2 = get_var(img2)
    for i in range(x):
        for j in range(y):
            # 除去所有的分母为0的情况
            if var1[i, j] + var2[i, j] == 0:
                var1[i, j] = var2[i, j] = 0
                continue
            w1 = var1[i, j] / (var1[i, j] + var2[i, j])
            w2 = var2[i, j] / (var1[i, j] + var2[i, j])
            # 通过权重平方，可以更凸显清楚的图像
            if w1 < w2:
                w1 = w1 ** 2
                w2 = 1 - w1
            else:
                w2 = w2 ** 2
                w1 = 1 - w2
            var1[i, j] = w1
            var2[i, j] = w2
    return var1, var2


# 图像融合
def merge3(img1, img2):
    # 拆分除bgr三色通道，挨个处理
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)
    # 得到每张图每个通道的权重
    b1_w, b2_w = get_pow(b1, b2)
    g1_w, g2_w = get_pow(g1, g2)
    r1_w, r2_w = get_pow(r1, r2)
    new_img = np.zeros_like(img1)
    x, y, z = new_img.shape
    nb, ng, nr = np.zeros_like(b1), np.zeros_like(g1), np.zeros_like(r1)
    for i in range(x):
        for j in range(y):
            # 权重融合
            nb[i, j] = (b1_w[i, j] * b1[i, j] + b2_w[i, j] * b2[i, j]).astype(int)
            ng[i, j] = (g1_w[i, j] * g1[i, j] + g2_w[i, j] * g2[i, j]).astype(int)
            nr[i, j] = (r1_w[i, j] * r1[i, j] + r2_w[i, j] * r2[i, j]).astype(int)
    # 合并为一张彩色图
    new_img = cv2.merge([nb, ng, nr])
    return new_img


# 图像融合
def merge4(img1, img2):
    # pepsi1.jpg   pepsi2.jpg
    # 这两张图是四通道图片，因此需要单独处理第四维 alpha
    b1, g1, r1, a1 = cv2.split(img1)
    b2, g2, r2, a2 = cv2.split(img2)
    # 得到每张图每个通道的权重
    b1_w, b2_w = get_pow(b1, b2)
    g1_w, g2_w = get_pow(g1, g2)
    r1_w, r2_w = get_pow(r1, r2)
    a1_w, a2_w = get_pow(a1, a2)
    new_img = np.zeros_like(img1)
    x, y, z = new_img.shape
    nb, ng, nr, na = np.zeros_like(b1), np.zeros_like(g1), np.zeros_like(r1), np.zeros_like(a1)
    for i in range(x):
        for j in range(y):
            # 权重融合
            nb[i, j] = (b1_w[i, j] * b1[i, j] + b2_w[i, j] * b2[i, j]).astype(int)
            ng[i, j] = (g1_w[i, j] * g1[i, j] + g2_w[i, j] * g2[i, j]).astype(int)
            nr[i, j] = (r1_w[i, j] * r1[i, j] + r2_w[i, j] * r2[i, j]).astype(int)
            na[i, j] = (a1_w[i, j] * a1[i, j] + a2_w[i, j] * a2[i, j]).astype(int)
    new_img = cv2.merge([nb, ng, nr])
    return new_img


def work():
    path1 = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/homework/1.jpg'
    path2 = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/homework/2.jpg'

    # 处理 IM000435 三通道图片使用merge3，用一下三行代码, 若使用该部分,注释掉下面部分
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img3 = merge3(img1, img2)

    # 处理 pepsi1 四通道图片使用merge4，用一下三行代码, 若使用该部分,注释掉上面部分
    # img1 = image.imread(path1)
    # img2 = image.imread(path2)
    # img3 = merge4(img1, img2)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img3)
    # cv2.imwrite('C:/Users/boliz/Desktop/homework/img1.jpg', img1)
    # cv2.imwrite('C:/Users/boliz/Desktop/homework/img2.jpg', img2)
    # cv2.imwrite('C:/Users/boliz/Desktop/homework/img3.jpg', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    work()