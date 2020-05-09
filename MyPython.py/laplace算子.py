import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt


def trans(x, _max=255):
    x = min(x, _max)
    x = max(x, 0)
    return x


def laplace2(img, c=0.3):
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
    # load image
    path = 'C:/Users/boliz/Desktop/Learning/Mathematical model calculation/第1次 作业/素材/nYvEfVn.jpg'
    img = np.array(image.imread(path))
    new_img, grad2_img = laplace2(img)

    fig, ax = plt.subplots(1, 3, figsize=(30, 64))
    ax[0].imshow(img)
    ax[1].imshow(grad2_img)
    ax[2].imshow(new_img)
    plt.show()
    return


if __name__ == '__main__':
    work(3)