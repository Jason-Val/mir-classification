import numpy as np
import fma_pca
import matplotlib.pyplot as plt
import datetime
import logger

num_classes = 2

subset='small'

fma_pca.init(num_classes, pca_on=True, subset=subset)

#num_classes = fma_pca.num_genres()

###############################################################################

train_x = fma_pca.x_train()
train_y = fma_pca.y_train()
test_x = fma_pca.x_test()
test_y = fma_pca.y_test()
val_x = fma_pca.x_val()
val_y = fma_pca.y_val()

num_test_tracks = len(test_x)
feature_size = len(train_x[0])

###############################################################################

import tensorflow as tf
#from sklearn.metrics import precision_recall_fscore_support

###############################################################################

test_id = 'batch_size/1-10-c2'      # for the log file name
training_epochs = 100000
batch_size = 10
minutes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
n_dim = feature_size
n_classes = num_classes
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.08

###############################################################################

X = tf.placeholder(tf.float32,[None,n_dim], name='X')
y_ = tf.placeholder(tf.float32,[None,n_classes] ,name='Y')

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name='W_1')
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name='b_1')
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1, name="h_1")


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name='W_2')
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name='b_2')
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2, name='h_2')


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name='W')
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name='b')
y = tf.matmul(h_2, W) + b
final_output = tf.nn.sigmoid(y, name='final_output')

init = tf.global_variables_initializer()

###############################################################################

#TODO: implement this by hand --- it might work better
cost_function = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

cost_function_manual = -tf.reduce_sum( (  (y_*tf.log(y + 1e-9)) + ((1-y_) * tf.log(1 - y + 1e-9)) ))

#cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

equal = tf.equal(tf.round(final_output), y_)
correct_pred = tf.reduce_min(tf.round(tf.cast(equal, tf.float32)), 1)
accuracy = tf.reduce_mean(correct_pred)

###############################################################################

#sess_i = tf.InteractiveSession()

y_true, y_pred = None, None
cost_hist = np.empty(shape=[0],dtype=float)
val_hist = np.empty(shape=[0],dtype=float)
with tf.Session() as sess:
    sess.run(init)
    d1 = datetime.datetime.now()
    epochs = 0
    for epoch in range(training_epochs):
    #while (True):
        print("Epoch: {}".format(epoch), sep=' ', end='\r', flush=True)
        if batch_size <= 0:
            _,cost = sess.run([optimizer, cost_function], feed_dict={X:train_x, y_:train_y})
            cost_hist = np.append(cost_hist, cost)
            valid = sess.run(accuracy, feed_dict={X:val_x, y_:val_y})
            val_hist = np.append(val_hist, valid)
        else:
            fma_pca.init_batch()
            valid = 0
            while(True):
                train_x, train_y = fma_pca.get_batch(batch_size)
                if len(train_x) == 0:
                   break
                _,cost = sess.run([optimizer, cost_function], feed_dict={X:train_x, y_:train_y})
                cost_hist = np.append(cost_hist, cost)
                valid = sess.run(accuracy, feed_dict={X:val_x, y_:val_y})
                val_hist = np.append(val_hist, valid)
        epochs += 1
        if (datetime.datetime.now() - d1) > datetime.timedelta(0, 60*minutes, 0):
            break
        if valid >= .9:
            break
    d2=datetime.datetime.now()
    print("Epoch: ", epochs)
    print("\n-------------------------------------------------")
    print("elapsed time: ", d2-d1)
    #saver = tf.train.Saver()
    #saver.save(sess, "./models/akshay")
    
    test_accuracy = sess.run(accuracy, feed_dict={X:test_x, y_:test_y})
    print("testing accuracy:")
    print(test_accuracy)
    
    train_x = fma_pca.x_train()
    train_y = fma_pca.y_train()
    train_accuracy = sess.run(accuracy, feed_dict={X:train_x, y_:train_y})
    print("training accuracy:")
    print(train_accuracy)

    logger.write_log('{}'.format(test_id), n_classes, epochs, batch_size, d2-d1, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    # run on training data to test over-fitting hypothesis
    
    y_pred = sess.run(tf.round(final_output), feed_dict={X: test_x})
    y_true = sess.run(y_, feed_dict={y_: test_y})
    
    
    print(y_pred[5])
    print(y_true[5])
    for i in range(len(y_pred[5])):
        if y_pred[5][i] != y_true[5][i]:
            print("Index {}, {} != {}".format(i, y_pred[5][i], y_true[5][i]))
    print('---------')

    print(y_pred[50])
    print(y_true[50])
    for i in range(len(y_pred[50])):
        if y_pred[50][i] != y_true[50][i]:
            print("Index {}, {} != {}".format(i, y_pred[50][i], y_true[50][i]))
    print('---------')
    
    print(y_pred[2])
    print(y_true[2])
    for i in range(len(y_pred[2])):
        if y_pred[2][i] != y_true[2][i]:
            print("Index {}, {} != {}".format(i, y_pred[2][i], y_true[2][i]))
    print('---------')

    print(y_pred[0])
    print(y_true[0])
    for i in range(len(y_pred[0])):
        if y_pred[0][i] != y_true[0][i]:
            print("Index {}, {} != {}".format(i, y_pred[0][i], y_true[0][i]))
    
    print('---------')
    print(y_pred[57])
    print(y_true[57])
    for i in range(len(y_pred[57])):
        if y_pred[57][i] != y_true[57][i]:
            print("Index {}, {} != {}".format(i, y_pred[57][i], y_true[57][i]))

###############################################################################

plt.plot(cost_hist)
plt.plot(val_hist)
plt.show()

#fig = plt.figure(figsize=(10,8))
#plt.plot(cost_history)
#plt.ylabel("Cost")
#plt.xlabel("Iterations")
#plt.axis([0,training_epochs,0,np.max(cost_history)])
#plt.show()

#p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
#print ("F-Score:", round(f,3))
