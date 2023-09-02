import numpy as np
import tensorflow as tf


class FIDScorer:
    def __init__(self):
        self.InceptionModel = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", pooling="avg"
        )
