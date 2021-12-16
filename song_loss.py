from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.losses import Loss

rng = np.random.default_rng(12345)


def rmse(y_true, y_pred):  # difference between true label and predicted label
    error = y_true - y_pred  # square of the error
    sqr_error = backend.square(error)  # mean of the square of the error
    mean_sqr_error = backend.mean(sqr_error)
    return mean_sqr_error  # applying the loss function


def centroid_norm(
    samples, centroids
):  # difference between true label and predicted label
    error = samples - centroids  # square of the error
    sqr_error = backend.square(error)  # mean of the square of the error
    sqr_error = backend.mean(sqr_error)
    return sqr_error  # applying the loss function


class Centroid:
    """Define the controid of a cluster."""

    def __init__(self, number: int, feature_size: int, encoder) -> None:
        self.encoder = encoder
        self.number = number
        self.centroids = [np.zeros((feature_size)) for _ in range(number)]
        self.feature_size = feature_size
        self.associated_features = [0 for _ in range(number)]
        # Dict containing as a key the ref of each sample tensor and it's assignement.add()
        # It gets populated during the computation
        self.assignments = defaultdict(lambda: np.random.randint(0, 10))
        # To know if we are at the correct step or if it needs an assignement
        self.step = defaultdict(lambda: 1)
        self.current_step = 0

    @staticmethod
    def _get_indexes_of_samples(batch_number: int, batch_size: int) -> List[int]:
        start = batch_number * batch_size
        return list(range(start, start + batch_size))

    def get_assigned_centroid(self, sample_index: int) -> np.ndarray:
        centroids = np.array(self.centroids)
        return centroids[self.assignments[sample_index]]

    def get_assigned_centroids(
        self, nb_of_samples: int, batch_number: int
    ) -> np.ndarray:
        indexes = Centroid._get_indexes_of_samples(batch_number, nb_of_samples)
        assigned_centroids = np.array(itemgetter(*indexes)(self.assignments))
        centroids = np.array(self.centroids)
        return centroids[assigned_centroids]

    def get_closest_centroid(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """Return the closest centroid to each feature in [features] using the L2 norm.

        Args:
            features (np.ndarray): Input feature in feature space

        Returns:
            Tuple[np.ndarray, List[int]]: Array describing the closest centroid,
            and list of centroids indexes.
        """
        features_encoded = self.encoder(features).numpy()
        assert len(features_encoded.shape) == 2
        indexes = []
        centroids = np.array(self.centroids)
        for feature in features_encoded:
            dists = np.linalg.norm(centroids - feature, axis=1)
            w = np.where(dists == np.min(dists))
            indexes.append(w[0][0])
        return centroids[indexes], indexes

    def update_assignments(self, samples: tf.Tensor, batch_number: int):
        """Update [centroids] assignement for every sample of [samples].

        Args:
            samples (tf.Tensor): the samples to update the assigment for.
            batch_number (int): the index of the batch compared to the full dataset.
        """

        # Find index of each sample
        indexes = Centroid._get_indexes_of_samples(batch_number, len(samples))
        # Encode them
        encoded_samples = self.encoder(samples)
        centroids = np.array(self.centroids)
        # for each sample, we update the centroid assignement.
        for count, encoded_sample in enumerate(encoded_samples):
            sample_index = indexes[count]
            if self.step[sample_index] < self.current_step:
                dists = np.linalg.norm(centroids - encoded_sample, axis=1)
                index_assigned_centroid = np.where(dists == np.min(dists))[0][0]
                self.assignments[sample_index] = index_assigned_centroid
                self.step[sample_index] = self.current_step

    def update_centroids(self):
        for count, _ in enumerate(self.centroids):
            if self.associated_features[count] != 0:
                self.centroids[count] /= self.associated_features[count]
            else:
                print("No associated features for centroid", count)
        self.current_step += 1

    def init(self):
        self.associated_features = [0 for _ in range(self.number)]
        self.centroids = [np.zeros((self.feature_size)) for _ in range(self.number)]

    def sum_centroids(self, samples: tf.Tensor, batch_number: int):
        """Does a partial update of the [centroids] from [samples]. Each sample
        in [samples] in encoded and summed in its assigned centroid.

        once all samples have been summed in the [centroids], one must call
        self.update_centroids()

        Args:
            samples (tf.Tensor): a batch of semples
            batch_number (int): the index of the current batch in the dataset
        """
        # Get all the assignement. If the assignment does not exist it is randomly assigned.
        indexes = Centroid._get_indexes_of_samples(batch_number, len(samples))
        encoded_samples = self.encoder(samples).numpy()
        assigned_centroids = np.array(itemgetter(*indexes)(self.assignments))
        for count, _ in enumerate(self.centroids):
            mask = assigned_centroids == count
            masked_samples = encoded_samples[mask, :]
            sum_array = np.sum(masked_samples, axis=0)
            self.centroids[count] = self.centroids[count] + sum_array
            self.associated_features[count] += len(masked_samples)

    def init_centroids_at_samples(self, samples: np.ndarray):
        encoded_samples = self.encoder(samples).numpy()
        self.centroids = np.array(encoded_samples[0 : self.number, :]).tolist()


class LossClusteringSong(Loss):
    """
    Loss function as described by Song et. al (2014).

    Use the fact that y_true is the same x_input because we have a auto-encoder."""

    def __init__(
        self,
        encoder: tf.keras.Sequential,
        lambda_t: float,
        feature_size: int,
        batch_size: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.lambda_t = lambda_t
        self.centroids = Centroid(10, feature_size, self.encoder)
        self.batch_size = batch_size

    def init_centroids(self):
        self.centroids.init()

    def update_centroids(self):
        self.centroids.update_centroids()

    def sum_centroids(self, samples: tf.Tensor, count: int):
        self.centroids.sum_centroids(samples, count)

    def update_assignments(self, samples, count: int):
        self.centroids.update_assignments(samples, count)

    def init_centroids_at_samples(self, samples):
        self.centroids.init_centroids_at_samples(samples)

    def mapping_loss(self, y_true, batch_nb):
        centroids = self.centroids.get_assigned_centroids(len(y_true), batch_nb)
        f_x = self.encoder(y_true)
        loss = self.lambda_t * centroid_norm(f_x, centroids)
        return loss

    # compute loss
    def call(self, y_true, y_pred, batch_nb):
        """y_true correspond has the size of the output, so here it would be 100 features * batch_size"""
        the_loss = rmse(y_true, y_pred)
        if self.lambda_t > 0:
            mapping_loss = self.mapping_loss(y_true, batch_nb)
            # print("losses:", the_loss, "mapping", mapping_loss)
            the_loss += mapping_loss
            # print("final loss:", the_loss)
        return the_loss
