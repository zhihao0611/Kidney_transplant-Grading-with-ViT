import tensorflow as tf
import numpy as np
from keras import backend as K


def smooth_labeling(labels, dtype, label_smoothing):
    labels = tf.cast(labels, dtype=dtype)
    labels = (1 - label_smoothing) * labels + 0.5 * label_smoothing
    return labels


class categorical_focal_loss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=[[0.25, 0.25]],
        gamma=2.0,
        smoothing=0.1,
        max=1.0 - 1e-7,
        min=1e-7,
        name="Focalloss",
        from_logits=False,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = np.array(alpha, dtype=np.float32)
        self.gamma = gamma
        self.smoothing = smoothing
        self.max = max
        self.min = min
        self.from_logits = from_logits

    def call(self, labels, p):
        # Apply label smoothing
        labels = smooth_labeling(
            labels, dtype=K.floatx(), label_smoothing=self.smoothing
        )

        # For numerical stability
        p = tf.clip_by_value(p, self.min, self.max)

        # from_logits or not
        if self.from_logits:
            p = tf.nn.softmax(p)

        # Calculate cross_entropy
        cross_entropy = -labels * K.log(p)

        # Calculate Focal loss
        loss = self.alpha * K.pow(1 - p, self.gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "smoothing": self.smoothing,
            "max": self.max,
            "min": self.min,
            "from_logits": self.from_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}
