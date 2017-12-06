import numpy as np
import tensorflow as tf
import fma_pca

fma_pca.init()
feature_size = fma_pca.max_length()
num_genres = fma_pca.num_genres()

feature_column = [tf.feature_column.numeric_column("x", shape=[1])]

# check this
estimator = tf.estimator.Classifier("""....""")


# our input data.
# None indicates the first dimension can be any length -- this lets us train
# on however many tracks we want. The second dimension is the features for a
# given song
x = tf.placeholder(tf.float32, [None, feature_size])

W = tf.Variable(tf.zeros(feature_size, num_genres))
b = tf.Variable(tf.zeros([num_genres]))

# Wx + b = y
y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, num_genres])

train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = fma_pca.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys}
