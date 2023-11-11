import cv2
import numpy as np
import imageio


class ImageMatrix:
    def __init__(self, images_name, img_width, img_height):
        self.images_name = images_name
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = (img_width * img_height)

    def get_matrix(self):
        # print("ImageMatrix self.img_height: ", self.img_height)
        # print("ImageMatrix self.img_width: ", self.img_width)
        col = len(self.images_name)
        img_mat = np.zeros((self.img_size, col))

        i = 0
        for name in self.images_name:
            # print("ImageMatrix name: ", name)
            gray_img = cv2.imread(name, 0)
            # print("ImageMatrix gray: ", gray_img)
            gray_img = cv2.resize(gray_img, (self.img_height, self.img_width))
            mat = np.asmatrix(gray_img)
            img_mat[:, i] = mat.ravel()
            i += 1
        return img_mat
