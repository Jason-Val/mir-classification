import numpy as np
import tensorflow as tf

# An arbitrary neural net
class Net:
    def __init__(self, optimizer, output, cost_function, X, y_, model_name):
        self.optimizer=optimizer
        self.output=output
        self.cost_function=cost_function
        self.X=X
        self.y_=y_
        self.model_name=model_name
        

###############################################################################
# ------ BUILD THE NET ------ #
def build_net(n_dim, n_classes, learning_rate):
    # ------ NET STRUCTURE ------ #

    model_name = 'akshaynet'
    
    n_hidden_units_one = 280
    n_hidden_units_two = 300

    sd = 1 / np.sqrt(n_dim)
    
    X = tf.placeholder(tf.float32,[None,n_dim], name='X')
    y_ = tf.placeholder(tf.float32,[None,n_classes], name='Y')

    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name='W_1', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name='b_1', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1, name="h_1")


    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name='W_2', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name='b_2', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2, name='h_2')


    W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name='W', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name='b', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    y = tf.matmul(h_2, W) + b
    final_output = tf.nn.sigmoid(y, name='final_output')

    # ------ ACCURACY AND LOSS ------ #
    
    #TODO: implement this by hand --- it might work better
    cost_function = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y), name='cost_function')

    #regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W)

    #cost_function = tf.reduce_mean((cost_function + beta * regularizer))
    
    """
    cost_function_manual = -tf.reduce_sum( (  (y_*tf.log(y + 1e-9)) + ((1-y_) * tf.log(1 - y + 1e-9)) ))
    """
    #cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    return Net(optimizer, final_output, cost_function, X, y_, model_name)
    #return (final_output, optimizer, cost_function, init)
