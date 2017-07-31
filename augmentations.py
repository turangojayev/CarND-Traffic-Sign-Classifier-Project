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
    def __init__(self, columns, rows, stddev_rotation_angle=15, **kwargs):
        super(Rotator, self).__init__(partial(rotate, columns=columns, rows=rows), **kwargs)
        self._stddev_rotation_angle = stddev_rotation_angle

    def _get_augmentation_params(self, data, size):
        degrees = np.random.normal(scale=self._stddev_rotation_angle, size=size)
        return zip(data, degrees)


class HistogramEqualizer(Augmentation):
    def __init__(self):
        super(HistogramEqualizer, self).__init__(equalize_histogram)

    def _get_augmentation_params(self, data, size):
        return data


class Squeezer(Augmentation):
    def __init__(self, columns, rows, stddev_horizontal_scale_coef=0.12, stddev_vertical_scale_coef=0.12, **kwargs):
        super(Squeezer, self).__init__(partial(squeeze_from_sides, columns=columns, rows=rows), **kwargs)
        self._horizontal_scale_stddev = stddev_horizontal_scale_coef
        self._vertical_scale_stddev = stddev_vertical_scale_coef

    def _get_augmentation_params(self, data, size):
        _, columns, rows, _ = data.shape

        def get_transformation_matrix():
            # initial three points
            points1 = np.float32([[0, 0], [0, rows], [columns, rows]])
            # points1 = np.float32([[columns / 2, 0], [0, rows / 2], [columns, rows / 2]])

            # the points after the transformation
            # don't take absolute value, to allow "zoom in"(perspective) type of transformations
            points2 = np.float32(
                [
                    [
                        columns * np.random.normal(scale=self._horizontal_scale_stddev, size=1)[0],
                        rows * np.random.normal(scale=self._vertical_scale_stddev, size=1)[0]
                    ],
                    [
                        columns * np.random.normal(scale=self._horizontal_scale_stddev, size=1)[0],
                        rows * (1 - np.random.normal(scale=self._vertical_scale_stddev, size=1)[0])
                    ],
                    [
                        columns * (1 - np.random.normal(scale=self._horizontal_scale_stddev, size=1)[0]),
                        rows * (1 - np.random.normal(scale=self._vertical_scale_stddev, size=1)[0])
                    ]
                ])
            # points2 = np.float32(
            #     [
            #         [columns / 2, 0],
            #         [columns * np.abs(np.random.normal(scale=self._scale_stddev, size=1))[0], rows / 2],
            #         [columns * 1 - np.abs(np.random.normal(scale=self._scale_stddev, size=1))[0], rows / 2]
            #     ])

            return cv2.getAffineTransform(points1, points2)

        transformation_matrices = [get_transformation_matrix() for _ in range(size)]
        return zip(data, transformation_matrices)


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
