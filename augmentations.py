from functools import partial

import cv2
import numpy as np

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))


def _apply(operation, params):
    if len(params) == 2:
        return operation(*params)
    else:
        return operation(params)


class Augmentation:
    def __init__(self, operation, with_replacement=False):
        self._operation = operation
        self._with_replacement = with_replacement

    def __call__(self, X, y, size=100):
        random_indices = np.random.choice(len(X), size=size, replace=self._with_replacement)
        augmentation_params = self._get_augmentation_params(X[random_indices], size)
        augmented = list(map(partial(_apply, self._operation), augmentation_params))
        X_additional = np.array(augmented).reshape(size, *X.shape[1:])
        X = np.vstack((X, X_additional))
        y = np.concatenate((y, y[random_indices]))
        return X, y, random_indices

    def _get_augmentation_params(self, data, size):
        pass


def add_noise(image, prob_distr):
    return image + prob_distr(size=image.shape)


class NoiseAdder(Augmentation):
    def __init__(self, prob_distr, **kwargs):
        super(NoiseAdder, self).__init__(partial(add_noise, prob_distr=prob_distr), **kwargs)

    def _get_augmentation_params(self, data, size):
        return data


class Rotator(Augmentation):
    def __init__(self, prob_distr, **kwargs):
        super(Rotator, self).__init__(rotate, **kwargs)
        self._prob_distr = prob_distr

    def _get_augmentation_params(self, data, size):
        degrees = self._prob_distr(size=size)
        return zip(data, degrees)


class Squeezer(Augmentation):
    def __init__(self, prob_distr, **kwargs):
        super(Squeezer, self).__init__(squeeze_from_sides, **kwargs)
        self._prob_distr = prob_distr

    def _get_augmentation_params(self, data, size):
        _, columns, rows, _ = data.shape

        def get_transformation_matrix():
            # initial three points
            points1 = np.float32([[0, 0], [0, rows], [columns, rows]])
            points2 = np.float32(
                [
                    [
                        columns * self._prob_distr(size=1)[0],
                        rows * self._prob_distr(size=1)[0]
                    ],
                    [
                        columns * self._prob_distr(size=1)[0],
                        rows * (1 - self._prob_distr(size=1)[0])
                    ],
                    [
                        columns * (1 - self._prob_distr(size=1)[0]),
                        rows * (1 - self._prob_distr(size=1)[0])
                    ]
                ])

            return cv2.getAffineTransform(points1, points2)

        transformation_matrices = [get_transformation_matrix() for _ in range(size)]
        return zip(data, transformation_matrices)


def gcn(img):
    mean = np.mean(img)
    lambbda = 10
    denominator = np.sqrt(lambbda + np.var(img))
    return (img - mean) / max(0.0001, denominator)


def squeeze_from_sides(image, transformation_matrix):
    rows, columns, *_ = image.shape
    return cv2.warpAffine(image, transformation_matrix, (columns, rows))


class Preprocessing:
    def __init__(self, operation):
        self._operation = operation

    def __call__(self, X):
        preprocessed = list(map(self._operation, X))
        return np.array(preprocessed).reshape(-1, *X.shape[1:])


class HistogramEqualizer(Preprocessing):
    def __init__(self):
        super(HistogramEqualizer, self).__init__(equalize_histogram)


class ContrastNormalization(Preprocessing):
    def __init__(self):
        super(ContrastNormalization, self).__init__(contrast_normalization)



class StandardScaling(ContrastNormalization):
    def __init__(self):
        super(StandardScaling, self).__init__()
        self._fit = False

    def __call__(self, X):
        if not self._fit:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._fit = True

        X = (X - self._mean) / self._std
        return super(StandardScaling, self).__call__(X)


def equalize_histogram(image):
    return CLAHE.apply(image)


def rotate(image, degree):
    rows, columns, *_ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), degree, 1)
    return cv2.warpAffine(image, rotation_matrix, (columns, rows))


def makeGaussian(size, sigma=2):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    gaussian = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
    gaussian /= np.sum(gaussian)
    return gaussian


gaussian = makeGaussian(5, 3)


def contrast_normalization(img):
    img1 = img.reshape(*img.shape[:2])
    subtractive_normalized = img1 - cv2.filter2D(img1, 3, gaussian)
    image = subtractive_normalized ** 2
    height, width, *_ = image.shape

    output = np.sqrt(cv2.filter2D(image, 3, gaussian))
    mean_sigma = np.mean(output)
    indices = output < mean_sigma
    output[indices] = mean_sigma

    output = subtractive_normalized / output
    return output
