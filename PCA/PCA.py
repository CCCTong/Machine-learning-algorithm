import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image
from sklearn.decomposition import PCA


def work():
    path = "C:/Users/boliz/Desktop/Learning/Machine Learning/Homework/实验1_image_compression_SVD/butterfly.bmp"
    img = Image.open(path)
    data = np.array(img) / 255
    x, y, z = data.shape
    data = data.reshape(x, y*z)
    img_arr = np.array([10])
    for i in range(img_arr.shape[0]):
        eig_nums = img_arr[i]
        pca = PCA(n_components=eig_nums).fit(data)
        new_data = pca.transform(data)
        new_img = pca.inverse_transform(new_data)
        new_img = new_img.reshape(x, y, z) * 255
        image.imsave('C:/Users/boliz/Desktop/fig' + str(eig_nums) + ".png", new_img.astype(np.uint8))
    return


if __name__ == '__main__':
    work()
