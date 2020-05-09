import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from PIL import Image


def trans(x, _max=255):
    x = min(x, _max)
    x = max(x, 0)
    return x


def solve(img, c, gama):
   x, y = img.shape
   new_img = np.zeros_like(img)
   for i in range(x):
       for j in range(y):
           new_img[i][j] = trans(c * ((img[i][j] / 255) ** gama) * 255)
   return new_img


if __name__ == '__main__':
    # 修改为系统中图片路径
    path = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/第1次 作业/素材/'
    # 图片名
    img_path = 'img8.jpg'
    img = np.array(Image.open(path + img_path).convert("L"))
    new_img = solve(img, 1, 2.5)
    fig, ax = plt.subplots(1, 2, figsize=(30, 64))
    ax[0].imshow(img, plt.gray())
    ax[1].imshow(new_img, plt.gray())
    plt.show()
    image.imsave('C:/Users/boliz/Desktop/img10.png', new_img)
