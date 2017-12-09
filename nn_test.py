import numpy as np
import tensorflow as tf
import fma_pca

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

fma_pca.init()
feature_size = fma_pca.max_length()
num_genres = fma_pca.num_genres()

# feature_column = [tf.feature_column.numeric_column("x", shape=[1])]

# check this
# estimator = tf.estimator.Classifier("""....""")


# our input data.
# None indicates the first dimension can be any length -- this lets us train
# on however many tracks we want. The second dimension is the features for a
# given song
x = tf.placeholder(tf.float32, [None, feature_size])
y_ = tf.placeholder(tf.float32, [None, num_genres])

W = tf.Variable(tf.zeros(feature_size, num_genres))
b = tf.Variable(tf.zeros([num_genres]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Wx + b = y
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)

for _ in range(1000):
    batch_xs, batch_ys = fma_pca.next_batch(100)
    if len(batch_xs == 0):
        break
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys}

print(sess.run( accuracy, feed_dict={x: fma_pca.test_pca_x(), y_: fma_pca.test_pca_y()}))
