import os
import numpy as np
import pandas as pd
from libsvm.svmutil import *
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize


def main():
    dir_path_image = '/heart'
    types = ['heart', 'non-heart']
    train_data = []
    test_data = []
    target_data = []
    print(os.listdir('heart/train'))

    for i in types:
        for img in os.listdir(os.path.join(i, 'train')):
            if (img.split('.')[1] == 'jpg'):
                print(img)


main()
