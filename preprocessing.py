import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

with open('data/train.p', mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
rows, columns, channels = X_train[0].shape

# Points for defining the affine transformation that squeezes image from left and right
POINTS1 = np.float32([[columns / 2, 0], [0, rows / 2], [columns, rows / 2]])
POINTS2 = np.float32([[columns / 2, 0], [columns / 4, rows / 2], [columns * 3 / 4, rows / 2]])
SQUEEZE = cv2.getAffineTransform(POINTS1, POINTS2)

# Contrast Limited Adaptive Histogram Equalization
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))


def squeeze_from_sides(image):
    return cv2.warpAffine(image, SQUEEZE, (columns, rows))


def normalize(image):
    return (image - 128) / 128


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def equalize_histogram(image):
    return CLAHE.apply(image)


def rotate(image, degree):
    rotation_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), degree, 1)
    return cv2.warpAffine(image, rotation_matrix, (columns, rows))


# normalize and feed to cnns for 3 channels
# create other features for grayscale and feed to different branch of network

def plot(image):
    plt.imshow(image)
    plt.show()

    grayscaled = grayscale(image)
    squeezed = squeeze_from_sides(grayscaled)

    equalized = equalize_histogram(squeezed)
    plt.imshow(equalized, cmap='gray')
    plt.show()

    plt.imshow(cv2.flip(equalized, 1))
    plt.show()

    plt.imshow(rotate(equalized, degree=10), cmap='gray')
    plt.show()

    th3 = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    plt.imshow(th3, cmap='gray')
    plt.show()


# translation
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))
#
# perspective transform
#
# normalized = normalize(X_train[23487])
# plt.imshow(normalized, cmap='gray')
# plt.show()

plot(X_train[28129])
plot(X_train[27418])
plot(X_train[15929])
plot(X_train[17182])
plot(X_train[8115])
plot(X_train[5643])
plot(X_train[12949])
plot(X_train[31571])
plot(X_train[31578])
plot(X_train[33848])
plot(X_train[26418])
plot(X_train[14618])
plot(X_train[18418])


def pca_aug(img):
    from numpy import cov

    reshaped = img.reshape(3, -1)
    reshaped = (reshaped - np.mean(reshaped, axis=0)) / np.std(reshaped, axis=0)
    # reshaped = (reshaped - 128) / 128
    plt.imshow(reshaped.reshape(32, 32, 3))
    plt.show()
    # values, vectors = eigh(cov(reshaped))
    values, vectors = eig(cov(reshaped))
    print(values)

    pca = np.sqrt(values) * vectors
    perturb = (pca * np.random.randn(3) * 0.1).sum(axis=1)
    result = reshaped + perturb[:, None]
    print(result)
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    plt.imshow(result.reshape(32, 32, 3))
    plt.show()
    print(perturb)

# pca_aug(X_train[0])
# pca_aug(X_train[28129])
# pca_aug(X_train[27418])
# pca_aug(X_train[15929])
# pca_aug(X_train[17182])
# pca_aug(X_train[8115])
# pca_aug(X_train[5643])
# pca_aug(X_train[12949])
# pca_aug(X_train[31571])
# pca_aug(X_train[31578])
# pca_aug(X_train[33848])
# pca_aug(X_train[26418])
# pca_aug(X_train[14618])
# pca_aug(X_train[18418])
