from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from song_loss import LossClusteringSong


class AutoencoderSong(Model):
    """Standard auto-encoder in tensorflow"""

    def __init__(self, latent_dim, output_shape: Tuple[int, ...], threshold: float = 0):
        super().__init__()
        self.step = 0
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(1000, activation="relu"),
                layers.Dense(250, activation="relu"),
                layers.Dense(50, activation="relu"),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.song_loss = LossClusteringSong(self.encoder, threshold, 10, 10)

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(50, activation="relu"),
                layers.Dense(250, activation="relu"),
                layers.Dense(1000, activation="relu"),
                layers.Dense(np.prod(output_shape), activation="sigmoid"),
                layers.Reshape(output_shape),
            ]
        )

    def freeze_decoder(self):
        self.decoder.trainable = False
        # I don't know if that is needed?
        for layer in self.decoder.layers:
            layer.trainable = False

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        # print("Calling at the step", self.step)
        loss = self.song_loss.call(inputs, decoded, self.step)
        self.step += 1
        self.add_loss(loss)

        return decoded
