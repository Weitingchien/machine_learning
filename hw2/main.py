import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from libsvm.svmutil import *
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.datasets import dump_svmlight_file


def display(cases, first_image, p_acc_RBF, p_acc_linear):
    color = ['#00FFFF', '#7FFFD4', '#66CDAA', '#AFEEEE', '#40E0D0', '#48D1CC']
    print(f"(a) linear => Accuracy: {p_acc_linear[0]} (400 * 3 dims)")
    print(f"(b) RBF => Accuracy: {p_acc_RBF[0]} (400 * 3 dims)")
    accuracy = [p_acc_linear[0], p_acc_RBF[0], 0, 0, 0, 0]
    accuracy_ndarray = np.asarray(accuracy)
    # plt.title(cases[0])
    # print(first_image)
    #image = Image.fromarray(first_image)
    #figure, axis = plt.subplots(2, 2)
    #axis[0, 0].set_title()
    # plt.imshow(image)
    plt.bar(cases, accuracy_ndarray, color=color)
    plt.text(0.05, 1, '123')
    plt.show()


def main():
    cases = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    types = ['heart', 'non-heart']
    first_image = []
    train_data = []
    test_data = []
    label_train_data = []
    label_test_data = []
    # print(os.listdir('heart/train'))

    # train dataset
    for i in types:
        for j in os.listdir(os.path.join(i, 'train')):
            if (j.split('.')[1] == 'jpg'):
                label_train_data.append(types.index(i))
                image_numpy_ndarray = imread(f'{i}/train/{j}')
                image_resize = resize(image_numpy_ndarray, (20, 20, 3))
                train_data.append(image_resize.flatten())

    # test dataset
    for i in types:
        for j in os.listdir(os.path.join(i, 'test')):
            if (j.split('.')[1] == 'jpg'):
                label_test_data.append(types.index(i))
                image_numpy_ndarray = imread(f'{i}/test/{j}')
                if(j == 'pos071.jpg'):
                    # print(image_numpy_ndarray)
                    first_image = image_numpy_ndarray
                image_resize = resize(image_numpy_ndarray, (20, 20, 3))
                test_data.append(image_resize.flatten())

    # print(len(first_image))
    #print(f"train_data len: {len(train_data)}")
    #print(f"test_data len: {len(test_data)}")

    np_train_data = np.array(train_data)
    np_train_target_data = np.array(label_train_data)
    np_test_data = np.array(test_data)
    np_test_target_data = np.array(label_test_data)

    train_df = pd.DataFrame(np_train_data)
    print(train_df.shape)
    test_df = pd.DataFrame(np_test_data)
    train_df['label'] = np_train_target_data
    test_df['label'] = np_test_target_data

    train_x = train_df.iloc[:, :-1]  # -1表示不要選到label那行
    train_y = train_df.iloc[:, -1]
    test_x = train_df.iloc[:, :-1]
    test_y = train_df.iloc[:, -1]

    dump_svmlight_file(X=train_x, y=train_y, f='train_data.dat',
                       zero_based=True)  # 轉成LibSVM格式
    dump_svmlight_file(X=test_x, y=test_y, f='test_data.dat', zero_based=True)

    train_y, train_x = svm_read_problem('train_data.dat')
    test_y, test_x = svm_read_problem('test_data.dat')

    # print(type(train_y))

    """
    options:
        -s svm types: 0=> C-SVC 1=> nu-SVC 2=> one-class SVM 3=> epsilon-SVR 4=> nu-SVR (default 0)
        -t(kernel type): 0=> linear 1=> polynomial 2=> radial basis 3=> sigmoid (default 2)
        -b probability_estimates : whether to train a model for probability estimates, 0 or 1 (default 0)
    """

    #param = svm_parameter()
    model_svm_RBF = svm_train(train_y, train_x, '-s 0 -t 2 -b 1')
    p_label_RBF, p_acc_RBF, p_val_RBF = svm_predict(
        test_y, test_x, model_svm_RBF)

    model_svm_linear = svm_train(train_y, train_x, '-s 0 -t 0 -b 1')
    p_label_linear, p_acc_linear, p_val_linear = svm_predict(
        test_y, test_x, model_svm_linear)

    print(f"RBF: {p_acc_RBF}")
    print(f"linear: {p_acc_linear}")
    display(cases, first_image, p_acc_RBF, p_acc_linear)


main()
