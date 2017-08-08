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


class Rotator(Augmentation):
    def __init__(self, columns, rows, prob_distr, **kwargs):
        super(Rotator, self).__init__(partial(rotate, columns=columns, rows=rows), **kwargs)
        self._prob_distr = prob_distr

    def _get_augmentation_params(self, data, size):
        # degrees = np.random.normal(scale=self._stddev_rotation_angle, size=size)
        degrees = self._prob_distr(size=size)
        return zip(data, degrees)


class HistogramEqualizer(Augmentation):
    def __init__(self):
        super(HistogramEqualizer, self).__init__(equalize_histogram)

    def _get_augmentation_params(self, data, size):
        return data


class Squeezer(Augmentation):
    def __init__(self, columns, rows, prob_distr, **kwargs):
        super(Squeezer, self).__init__(partial(squeeze_from_sides, columns=columns, rows=rows), **kwargs)
        self._prob_distr = prob_distr

    def _get_augmentation_params(self, data, size):
        _, columns, rows, _ = data.shape

        def get_transformation_matrix():
            # initial three points
            points1 = np.float32([[0, 0], [0, rows], [columns, rows]])

            # the points after the transformation
            # don't take absolute value, to allow "zoom in"(perspective) type of transformations
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
    return (img - mean) / max(0, denominator)


class ContrastNormalization(Augmentation):
    def __init__(self):
        super(ContrastNormalization, self).__init__(gcn)

    def _get_augmentation_params(self, data, size):
        return data


class Flipper(Augmentation):
    def __init__(self):
        super(Flipper, self).__init__(flip)

    def _get_augmentation_params(self, data, size):
        return data


class Translator(Augmentation):
    def __init__(self):
        super(Translator, self).__init__()

    def _get_augmentation_params(self, data, size):
        # TODO:implement
        pass


def flip(image):
    return cv2.flip(image, 1)


def squeeze_from_sides(image, transformation_matrix, columns, rows):
    return cv2.warpAffine(image, transformation_matrix, (columns, rows))


def equalize_histogram(image):
    return CLAHE.apply(image)


def rotate(image, degree, columns, rows):
    rotation_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), degree, 1)
    return cv2.warpAffine(image, rotation_matrix, (columns, rows))
