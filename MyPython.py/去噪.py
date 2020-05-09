import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from PIL import Image


def trans(x, _max=1):
    # jpg 结尾的图片_max修改为255
    x = min(x, _max)
    x = max(x, 0)
    return x


def _filter(img, n):
    # img 为图片，n为区域大小
    new_img = np.zeros_like(img)
    x, y = img.shape[0:2]
    step = int(n / 2)
    # mid value filter
    for i in range(step, x - step):
        for j in range(step, y - step):
            new_img[i][j] = trans(np.median(img[i - step:i + step + 1, j - step:j + step + 1]))
    return new_img


def laplace2(img, c=0.4):
    x, y, z = img.shape
    filt = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    new_img = np.zeros_like(img)
    grad2_img = np.zeros_like(img)
    for i in range(1, x-1):
        for j in range(1, y-1):
            for k in range(z):
                grad2_img[i][j][k] = trans((img[i-1:i+2, j-1:j+2, k:k+1] * filt).sum())
                new_img[i][j][k] = trans(img[i][j][k] - c * grad2_img[i][j][k])
    return new_img, grad2_img


def work(n):
    # 加载图片
    path = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/第1次 作业/素材/image1.png'
    img = np.array(image.imread(path))

    # Create new image
    new_img = _filter(img, n)
    final_img, grad2_img = laplace2(new_img)

    # Show image
    fig, ax = plt.subplots(1, 4, figsize=(30, 64))
    ax[0].imshow(img, plt.gray())
    ax[1].imshow(new_img, plt.gray())
    ax[2].imshow(grad2_img, plt.gray())
    ax[3].imshow(final_img, plt.gray())
    plt.show()
    return


if __name__ == '__main__':
    # 入口
    work(4)