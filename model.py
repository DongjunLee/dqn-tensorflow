
import tensorflow as tf


class MLPv1:

    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: None, learning_rate=0.001) -> None:
        state_length = X.get_shape().as_list()[1]
        self.X = tf.reshape(X, [-1, state_length])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        net = self.X
        net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        net = tf.layers.dense(net, self.num_classes)

        print(net)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)


class ConvNetv1:


    def __init__(self, X: tf.placeholder, num_classes: int, frame_size: int=1, learning_rate=0.001) -> None:
        self.X = tf.reshape(X, [-1, 128, frame_size])

        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def build_network(self) -> None:
        conv1 = tf.layers.conv1d(self.X, 32, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv1d(pool1, 64, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

        conv3 = tf.layers.conv1d(pool2, 128, kernel_size=3, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)
        pool3_flat = tf.reshape(pool3, [-1, 16 * 128])

        net = tf.layers.dense(pool3_flat, 512)
        net = tf.layers.dense(net, 128)
        net = tf.layers.dense(net, self.num_classes)

        self.inference = net
        self.predict = tf.argmax(self.inference, 1)

        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.loss = tf.losses.mean_squared_error(self.Y, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
