import numpy as np
import tensorflow as tf
#import svd_grad


# An arbitrary neural net
class Net:
    def __init__(self, optimizer, output, cost_function, X, y_, model_name, keep_prob, final_pool):
        self.optimizer=optimizer
        self.output=output
        self.cost_function=cost_function
        self.X=X
        self.y_=y_
        self.model_name=model_name
        self.keep_prob=keep_prob
        self.final_pool=final_pool
        

###############################################################################


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# ------ BUILD THE NET ------ #


def build_net(dim, n_classes, learning_rate, pca_on=False):
    # ------ NET STRUCTURE ------ #
    
    model_name = 'cnn_net'
    
    X = tf.placeholder(tf.float32,[None,dim[0],dim[1], 1], name='X')
    y_ = tf.placeholder(tf.float32,[None,n_classes], name='Y')
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)


    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    #########################################################

    # reshapes the data into a matrix which we can use PCA on
    # Original 16x16x64 matrix: a 3D box, where each of 64 layers is 16x16,
    #  and corresponds to a feature in the image.
    # I return a matrix where each column is a flattened feature!!!
    # So this is something PCA can be run on
    
    dim_in = 0
    
    if pca_on:
        h_pool3_2d = tf.reshape(h_pool3, [-1, 16*16, 64])
        h_pool3_cen = h_pool3_2d - tf.reduce_mean(h_pool3_2d, 1, keep_dims=True)
        
        h_pool3_S, h_pool3_U, _ = tf.svd(h_pool3_cen, full_matrices=True)
        
        h_pool3_S = tf.matrix_diag(h_pool3_S)
        
        padding = tf.constant([[0,0], [0, 256-64], [0, 256-64]])
        h_pool3_S = tf.pad(h_pool3_S, padding, "CONSTANT")
        
        h_pool3_pca = tf.matmul(h_pool3_U, h_pool3_S)
        
        dim_in = 64*64
    else:
        dim_in = 16*16*64
        
    #h_pool3_flat = tf.reshape(h_pool3 [-1, dim_in])
    h_pool3_flat = tf.contrib.layers.flatten(h_pool3)

    W_fc1 = weight_variable([dim_in, 1024])
    b_fc1 = bias_variable([1024])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, n_classes])
    b_fc2 = bias_variable([n_classes])
    
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    final_output = tf.nn.sigmoid(y)

    # ------ ACCURACY AND LOSS ------ #
    
    cost_function = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y), name='cost_function')

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

    return Net(optimizer, final_output, cost_function, X, y_, model_name, keep_prob, h_pool3)
