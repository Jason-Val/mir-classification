import numpy as np
import akshaynet
import fma_pca
import matplotlib.pyplot as plt
import datetime
import logger
import tensorflow as tf
import cnn2

def train_net(net, test_id, training_epochs, batch_size, minutes):
    
    y = tf.round(net.output)
    
    # accuracy computes whether or not the vectors match exactly
    equal = tf.equal(y, net.y_)
    correct_pred = tf.reduce_min(tf.round(tf.cast(equal, tf.float32)), 1)
    accuracy = tf.reduce_mean(correct_pred)
    
    # accuracy_lenient returns the fraction of 1s the net got correct, or 0 if it guessed even a single incorrect genre
    
    # calculates fraction of 1s in y_ that are matched in y
    count = tf.reduce_sum( tf.multiply(y, net.y_), 1)
    count_ = tf.reduce_sum(net.y_, 1)
    ratio = tf.divide(count, count_)
    
    # calculates the mask for removing incorrect guesses
    sub = tf.subtract(net.y_, y)
    mask = tf.cast(tf.reduce_min(sub, 1) + 1, dtype=tf.float32)
    
    # applies the mask and gets the average score
    stack = tf.stack([mask, ratio])    
    score_final = tf.reduce_min(stack, 1)
    accuracy_lenient = tf.reduce_mean(stack)
    
    

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
        stop = False
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
                    if (datetime.datetime.now() - d1) > datetime.timedelta(0, 60*minutes, 0):
                        stop = True                        
                        break
            epochs += 1
            if (datetime.datetime.now() - d1) > datetime.timedelta(0, 60*minutes, 0) or stop == True:
                break
            if valid >= 1:
                print("Sufficiently Valid!")
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

        test_accuracy2 = sess.run(accuracy_lenient, feed_dict={net.X:x_test, net.y_:y_test, net.keep_prob:1.0})
        print("testing accuracy lenient:")
        print(test_accuracy2)
        
        # Too much data in medium subset to run through without batching 
        """
        train_accuracy = sess.run(accuracy, feed_dict={net.X:x_train, net.y_:y_train, net.keep_prob:1.0})
        print("training accuracy:")
        print(train_accuracy)

        train_accuracy2 = sess.run(accuracy_lenient, feed_dict={net.X:x_train, net.y_:y_train, net.keep_prob:1.0})
        print("training accuracy lenient:")
        print(train_accuracy2)
        """
        
    return (test_accuracy, -1, cost_hist, val_hist, epochs, d2-d1)

def show_plot(cost_hist, val_hist):
    plt.plot(cost_hist)
    plt.plot(val_hist)
    plt.show()



def go():
    n_classes = 4
    subset='medium'
    
    fma_pca.init_mel(n_classes, subset=subset, reuse=True, pca_on=False)

    #fma_pca.init_mel(n_classes, subset=subset, reuse=True)
    
    test_id = 'Accuracy/5hr_'        # for the log file name
    training_epochs = 100000000
    batch_size = 700
    minutes = 5*60
    n_dim = fma_pca.n_dim()
    sd = 1 / np.sqrt(n_dim)
    learning_rate=.01
    
##############################################################################
    
    test_id1 = '{}cnn_c4_medium'.format(test_id)
    net = cnn2.build_net([128, 128], n_classes, learning_rate)
    
    test_accuracy, train_accuracy, cost_hist, val_hist, epochs, time = train_net(net, test_id1, training_epochs, batch_size, minutes)
    
    logger.write_log(test_id1, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost_hist, val_hist)
    
    show_plot(cost_hist, val_hist)
    show_plot(cost_hist, [])
    show_plot(val_hist, [])

if __name__ == '__main__':
    go()



