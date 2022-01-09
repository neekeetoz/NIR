import cv2
import imageio
import numpy as np
import os
import skimage

from os import listdir
from os.path import isfile, join

root = 'D:/Users/inet/Documents/GitHub/Nir_all/Nikita_K/GTSRB_Challenge/train'

output_train = 'D:/Users/inet/Documents/GitHub/Nir_all/Nikita_K/GTSRB_Challenge/train_pre'
output_test = 'D:/Users/inet/Documents/GitHub/Nir_all/Nikita_K/GTSRB_Challenge/test_pre'


def ppt2jpg(root):

    for dir_name in os.listdir(join(root)):

        c = []
        names = os.listdir(join(root, dir_name))

        for name in names:
            index = name.rfind(".")
            name = name[:index]
            c.append(name)
        for files in c:
            picture_path = join(root, dir_name, files + ".ppm")
            out_path = join(output_train, dir_name, dir_name + '_' + files + ".jpg")
            image = cv2.imread(picture_path)
            cv2.imwrite(out_path, image)
        print("all is changed")


def main():
    ppt2jpg(root)


if __name__ == "__main__":
    main()
