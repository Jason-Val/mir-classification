import os

"""
Writes logs for output of neural net training
"""
def write_log(name, n_classes, epochs, batch_size, time, learning_rate, subset, train_accuracy, test_accuracy, cost, valid):
    
    filename = './logs/{}'.format(name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open('{}.log'.format(filename), 'w') as f:
        f.write('Number of classes: {}\n'.format(n_classes))
        f.write('Elapsed time: {}\n'.format(time))
        if batch_size <= 0:
            f.write('One batch.\nNumber of epochs: {}\n'.format(epochs))
        else:
            f.write('Mini-batch size: {}\nNumber of epochs: {}\n'.format(batch_size, epochs))
        f.write('Learning rate: {}\n'.format(learning_rate))
        f.write('Subset: {}\n'.format(subset))
        f.write('------------------------------------\n')
        f.write('Testing accuracy: {}\n'.format(test_accuracy))
        f.write('Training accuracy: {}\n'.format(train_accuracy))
        f.write('------------------------------------\n'.format())
        f.write('cost history,\tvalidation history\n'.format())
        for c, v in zip(cost, valid):
            f.write('{},\t{}\n'.format(c, v))
