import numpy as np
import tensorflow as tf

class Net:
    def __init__(self, optimizer, output, cost_function, X, y_, model_name, keep_prob):
        self.optimizer=optimizer
        self.output=output
        self.cost_function=cost_function
        self.X=X
        self.y_=y_
        self.model_name=model_name
        self.keep_prob=keep_prob
        

###############################################################################
# ------ BUILD THE NET ------ #

def make_layer(l_in, shape_1, shape_2, sd):
    W = tf.Variable(tf.random_normal([shape_1, shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b = tf.Variable(tf.random_normal([shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    h = tf.nn.sigmoid(tf.matmul(l_in,W) + b)
    return h

def make_input(l_in, shape_1, shape_2, sd):
    W = tf.Variable(tf.random_normal([shape_1, shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b = tf.Variable(tf.random_normal([shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    h = tf.nn.tanh(tf.matmul(l_in,W) + b)
    return h

def make_output(l_in, shape_1, shape_2, sd):
    W = tf.Variable(tf.random_normal([shape_1, shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b = tf.Variable(tf.random_normal([shape_2], mean = 0, stddev=sd), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    h = tf.matmul(l_in,W) + b
    return h

def build_net(n_dim, n_classes, learning_rate):
    # ------ NET STRUCTURE ------ #

    model_name = 'akshaynet'
    
    n_hidden_units_one = 2048
    n_hidden_units_two = 1024
    n_hidden_units_three = 512
    
    keep_prob = tf.placeholder(tf.float32)

    sd = 1 / np.sqrt(n_dim)
    
    X = tf.placeholder(tf.float32,[None,n_dim], name='X')
    y_ = tf.placeholder(tf.float32,[None,n_classes], name='Y')

    l1 = make_input(X, n_dim, n_hidden_units_one, sd)
    l2 = make_layer(l1, n_hidden_units_one, n_hidden_units_two, sd)

    dropout1 = tf.nn.dropout(l2, keep_prob)

    l3 = make_layer(dropout1, n_hidden_units_two, n_hidden_units_three, sd)
    
    dropout2 = tf.nn.dropout(l3, keep_prob)

    y = make_output(dropout2, n_hidden_units_three, n_classes, sd)

    final_output = tf.nn.sigmoid(y, name='final_output')

    # ------ ACCURACY AND LOSS ------ #
    
    cost_function = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y), name='cost_function')
    
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

    return Net(optimizer, final_output, cost_function, X, y_, model_name, keep_prob)
