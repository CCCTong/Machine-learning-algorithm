import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def work():
    path = "C:/Users/boliz/Desktop/Learning/Machine Learning/Homework/实验1_image_compression_SVD/butterfly.bmp"
    img = mpimg.imread(path)
    x, y, z = img.shape
    img_temp = img.reshape(x, y * z)
    U, sigma, VT = np.linalg.svd(img_temp)

    img_arr = np.array([10, 25, 40, 55, 70, 85, 100, 130, 160, 190, 220, 243])
    for i in range(img_arr.shape[0]):
        sval_nums = img_arr[i]
        img2 = (U[:, 0:sval_nums]).dot(np.diag(sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
        img2 = img2.reshape(x, y, z)
        mpimg.imsave('C:/Users/boliz/Desktop/fig' + str(sval_nums) + ".png", img2.astype(np.uint8))
    return


if __name__ == '__main__':
    work()
