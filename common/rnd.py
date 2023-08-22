import tensorflow as tf

class Predictor(tf.keras.Model):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.Flatten()
        ])
        conv_output_size = self._get_conv_output(input_shape)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])

    def call(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def _get_conv_output(self, shape):
        x = tf.zeros((1, *shape))
        x = self.conv(x)
        return int(tf.reduce_prod(x.shape[1:]))

class RandomNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super(RandomNet, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.Flatten()
        ])
        conv_output_size = self._get_conv_output(input_shape)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim)
        ])

    def call(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def _get_conv_output(self, shape):
        x = tf.zeros((1, *shape))
        x = self.conv(x)
        return int(tf.reduce_prod(x.shape[1:]))
