import numpy as np
import akshaynet
import fma_pca
import matplotlib.pyplot as plt
import datetime
import logger
import tensorflow as tf

def train_net(net, test_id, training_epochs, batch_size, minutes):
    
    equal = tf.equal(tf.round(net.output), net.y_)
    correct_pred = tf.reduce_min(tf.round(tf.cast(equal, tf.float32)), 1)
    accuracy = tf.reduce_mean(correct_pred)

    x_train = fma_pca.x_train()
    y_train = fma_pca.y_train()
    x_test = fma_pca.x_test()
    y_test = fma_pca.y_test()
    x_val = fma_pca.x_val()
    y_val = fma_pca.y_val()
    
    init = tf.global_variables_initializer()
    
    cost_hist = np.empty(shape=[0],dtype=float)
    val_hist = np.empty(shape=[0],dtype=float)

    with tf.Session() as sess:
        sess.run(init)
        d1 = datetime.datetime.now()
        epochs = 0
        for epoch in range(training_epochs):
            print("Epoch: {}".format(epoch), sep=' ', end='\r', flush=True)
            if batch_size <= 0:
                _,cost = sess.run([net.optimizer, net.cost_function], feed_dict={net.X:x_train, net.y_:y_train})
                cost_hist = np.append(cost_hist, cost)
                valid = sess.run(accuracy, feed_dict={net.X:x_val, net.y_:y_val})
                val_hist = np.append(val_hist, valid)
            else:
                fma_pca.init_batch()
                valid = 0
                while(True):
                    x_batch, y_batch = fma_pca.get_batch(batch_size)
                    if len(x_batch) == 0:
                       break
                    _,cost = sess.run([net.optimizer, net.cost_function], feed_dict={net.X:x_batch, net.y_:y_batch})
                    cost_hist = np.append(cost_hist, cost)
                    valid = sess.run(accuracy, feed_dict={net.X:x_val, net.y_:y_val})
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
        
        if net.model_name != '':
            saver = tf.train.Saver()
            saver.save(sess, "./models/{}".format(net.model_name))
        
        test_accuracy = sess.run(accuracy, feed_dict={net.X:x_test, net.y_:y_test})
        print("testing accuracy:")
        print(test_accuracy)

        train_accuracy = sess.run(accuracy, feed_dict={net.X:x_train, net.y_:y_train})
        print("training accuracy:")
        print(train_accuracy)

        y_pred = sess.run(tf.round(final_output), feed_dict={X: x_test})
        y_true = sess.run(y_, feed_dict={y_: y_test})
        
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

    return (test_accuracy, train_accuracy, cost_hist, val_hist, epochs, d2-d1)

def show_plot(cost_hist, val_hist):
    plt.plot(cost_hist)
    plt.plot(val_hist)
    plt.show()


n_classes = 2
subset='small'

fma_pca.init(n_classes, pca_on=True, subset=subset, n_components=3, reuse=True)

test_id = 'misc/0'      # for the log file name
training_epochs = 100000000
batch_size = 150
minutes = 3
n_dim = fma_pca.n_dim()
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.17

net = akshaynet.build_net(n_dim, n_classes, learning_rate)

test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id, training_epochs, batch_size, minutes)

logger.write_log('{}'.format(test_id), n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)

show_plot(cost_hist, val_hist)
