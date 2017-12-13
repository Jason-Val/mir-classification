import numpy as np
import akshaynet
import fma_pca
import matplotlib.pyplot as plt
import datetime
import logger
import tensorflow as tf
import cnn2
import svd_grad
    
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
                _,cost = sess.run([net.optimizer, net.cost_function], feed_dict={net.X:x_train, net.y_:y_train, net.keep_prob:.5})
                cost_hist = np.append(cost_hist, cost)
                valid = sess.run(accuracy, feed_dict={net.X:x_val, net.y_:y_val, net.keep_prob:1.0})
                val_hist = np.append(val_hist, valid)
            else:
                fma_pca.init_batch()
                valid = 0
                while(True):
                    x_batch, y_batch = fma_pca.get_batch(batch_size)
                    if len(x_batch) == 0:
                       break
                    _,cost = sess.run([net.optimizer, net.cost_function], feed_dict={net.X:x_batch, net.y_:y_batch, net.keep_prob:.5})
                    cost_hist = np.append(cost_hist, cost)
                    valid = sess.run(accuracy, feed_dict={net.X:x_val, net.y_:y_val, net.keep_prob:1.0})
                    val_hist = np.append(val_hist, valid)
            epochs += 1
            if (datetime.datetime.now() - d1) > datetime.timedelta(0, 60*minutes, 0):
                break
            if valid >= 1:
                print("Sufficiently Valid!!!!")
                break
        d2=datetime.datetime.now()
        print("Epoch: ", epochs)
        print("\n-------------------------------------------------")
        print("elapsed time: ", d2-d1)
        
        if net.model_name != '':
            saver = tf.train.Saver()
            saver.save(sess, "./models/{}".format(net.model_name))
        
        test_accuracy = sess.run(accuracy, feed_dict={net.X:x_test, net.y_:y_test, net.keep_prob:1.0})
        print("testing accuracy:")
        print(test_accuracy)

        train_accuracy = sess.run(accuracy, feed_dict={net.X:x_train, net.y_:y_train, net.keep_prob:1.0})
        print("training accuracy:")
        print(train_accuracy)
        
        """
        y_pred = sess.run(tf.round(net.output), feed_dict={net.X: x_test, net.keep_prob:1})
        y_true = sess.run(net.y_, feed_dict={net.y_: y_test, net.keep_prob:1})
        
        print(y_pred[1])
        print(y_true[1])
        for i in range(len(y_pred[5])):
            if y_pred[5][i] != y_true[5][i]:
                print("Index {}, {} != {}".format(i, y_pred[5][i], y_true[5][i]))
        print('---------')

        print(y_pred[56])
        print(y_true[56])
        for i in range(len(y_pred[50])):
            if y_pred[50][i] != y_true[50][i]:
                print("Index {}, {} != {}".format(i, y_pred[50][i], y_true[50][i]))
        print('---------')
        
        print(y_pred[4])
        print(y_true[4])
        for i in range(len(y_pred[2])):
            if y_pred[2][i] != y_true[2][i]:
                print("Index {}, {} != {}".format(i, y_pred[2][i], y_true[2][i]))
        print('---------')

        print(y_pred[9])
        print(y_true[9])
        for i in range(len(y_pred[0])):
            if y_pred[0][i] != y_true[0][i]:
                print("Index {}, {} != {}".format(i, y_pred[0][i], y_true[0][i]))
        
        print('---------')
        print(y_pred[150])
        print(y_true[150])
        for i in range(len(y_pred[57])):
            if y_pred[57][i] != y_true[57][i]:
                print("Index {}, {} != {}".format(i, y_pred[57][i], y_true[57][i]))
        """

    return (test_accuracy, train_accuracy, cost_hist, val_hist, epochs, d2-d1)

def show_plot(cost_hist, val_hist):
    plt.plot(cost_hist)
    plt.plot(val_hist)
    plt.show()



def go():
    n_classes = 2
    subset='small'
    
    #fma_pca.init_mel(n_classes, subset=subset, reuse=True, pca_on=False)

    fma_pca.init(n_classes, subset=subset, reuse=True, pca_on=True)
    
    test_id = 'Presnetation/5min_'        # for the log file name
    training_epochs = 100000000
    batch_size = 400
    minutes = 5
    n_dim = fma_pca.n_dim()
    sd = 1 / np.sqrt(n_dim)
    learning_rate=.01
    
##############################################################################

    test_id1 = '{}nn_c2_pca_on'.format(test_id)
    # 128, 57 if pca, else 128,128
    net = akshaynet.build_net(n_dim, n_classes, learning_rate)
    
    test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id1, training_epochs, batch_size, minutes)
    
    logger.write_log(test_id1, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    show_plot(cost_hist, val_hist)
    show_plot(cost_hist, [])
    show_plot(val_hist, [])

##############################################################################

    fma_pca.init(n_classes, subset=subset, reuse=True, pca_on=False)

    test_id1 = '{}nn_c2_pca_off'.format(test_id)
    # 128, 57 if pca, else 128,128
    net = akshaynet.build_net(n_dim, n_classes, learning_rate)
    
    test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id1, training_epochs, batch_size, minutes)
    
    logger.write_log(test_id1, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    show_plot(cost_hist, val_hist)
    show_plot(cost_hist, [])
    show_plot(val_hist, [])

##############################################################################

    subset='small'
    fma_pca.init_mel(n_classes, subset=subset, reuse=False, pca_on=False)

##############################################################################
    
    test_id1 = '{}cnn_c2_pca_off'.format(test_id)
    # 128, 57 if pca, else 128,128
    net = cnn2.build_net([128,128], n_classes, learning_rate)
    
    test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id1, training_epochs, batch_size, minutes)
    
    logger.write_log(test_id1, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    show_plot(cost_hist, val_hist)
    show_plot(cost_hist, [])
    show_plot(val_hist, [])
    
##############################################################################

    fma_pca.init(n_classes, subset=subset, reuse=True, pca_on=True)

    test_id1 = '{}cnn_c2_pca_on'.format(test_id)
    # 128, 57 if pca, else 128,128
    net = cnn2.build_net([128,57], n_classes, learning_rate)
    
    test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id1, training_epochs, batch_size, minutes)
    
    logger.write_log(test_id1, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    show_plot(cost_hist, val_hist)
    show_plot(cost_hist, [])
    show_plot(val_hist, [])


if __name__ == '__main__':
    go()



