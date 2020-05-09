import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from PIL import Image


def trans(x, _max, _min):
    val = int(255/(_max-_min))*(x - _min)
    val = max(val, 0)
    val = min(val, 255)
    return val


def solve(img):
    x, y = img.shape
    _max = 200
    _min = 70
    new_img = np.zeros_like(img)
    for i in range(x):
        for j in range(y):
            new_img[i][j] = trans(img[i][j], _max, _min)
    return new_img


def pre_work(img):
    x, y = img.shape
    img_arr = img.reshape(x * y)
    plt.hist(x=img_arr, bins=50, color='steelblue', edgecolor='black')
    plt.show()


def work():
    # load image
    path = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/第1次 作业/素材//img6.jpg'
    img = np.array(Image.open(path).convert("L"))

    # init
    pre_work(img)

    # Create new image
    new_img = solve(img)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(30, 64))
    ax[0].imshow(img.astype(np.uint8), plt.gray())
    ax[0].set(title="src")
    ax[1].imshow(new_img.astype(np.uint8), plt.gray())
    ax[1].set(title="solve")
    plt.show()
    image.imsave('C:/Users/boliz/Desktop/fig1.png', new_img)
    return


if __name__ == '__main__':
    work()