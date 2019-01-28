import tensorflow as tf
import numpy as np
import pandas as pd
import fashion_data_import as fin
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder() # creating new encoder object

# logpath
LOG_PATH = '/home/wataru/machineLearn/kaggle/TF_practice/estimator/customEstimator/log'

# importing the fashion MNIST data from the script as numpy arrays
Xtrain, ytrain = fin.data_in('train')
Xtest, ytest = fin.data_in('test')
num_labels = np.unique(ytrain).size

# one hot matrix for labels
enc.fit(ytrain)
ytrain = enc.transform(ytrain).toarray()

enc.fit(ytest)
ytest = enc.transform(ytest).toarray()

# parameters
learning_rate = 0.001
batchSize = 500
LOGDIR = '/home/wataru/machineLearn/kaggle/TF_practice/estimator/customEstimator/log2'
num_iter = 50000


def model_fn(features, labels, mode, params):
    '''building NN'''

    # connecting first hidden layer to the input layer
    first_hidden_layer = tf.layers.dense(features['x'], 800, activation=tf.nn.relu)

    # connecting second hidden layer to the first hidden layer
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 800, activation=tf.nn.relu)

    # connecting third hidden layer to the second hidden layer
    third_hidden_layer = tf.layers.dense(second_hidden_layer, 400, activation=tf.nn.relu)

    # connecting output to the second hidden layer
    output_layer = tf.layers.dense(third_hidden_layer, num_labels, activation=tf.nn.relu)
    # shape of (batchSize x num_labels) oneHot, needs to be converted to dense
    

    # Reshape the output layer to dense vector(?, 2)
    predictions = output_layer #tf.where(tf.not_equal(output_layer, 0)) 
    print(predictions)
    # needs mod, return index??

    # provide ModeKey.PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions={'outfits': predictions})

    # calculate loss by mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)
    
    optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate']) # can also pass other params as necessary
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    ## tf.train.get_global_step get global step tensor. Whats' that??

    # calculate root mean squared error as additional eval metric
    # this metric needs to be accuracy % of the neural network
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                tf.cast(labels, tf.float64), predictions)} # data needs to be double to be

    # Provide an estimator spec for ModeKey.EVAL and TRAIN modes
    return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)


def main():

    ## defining input_fn functions for training and tesing
    
        
    train_in = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(Xtrain)},
            y=np.array(ytrain), num_epochs=None, shuffle=True, batch_size=batchSize)
    test_in = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(Xtest)},
            y=np.array(ytest), num_epochs=1, shuffle=False)
    
    # Model parameters
    model_params = {'learning_rate': learning_rate}
    
    # Instantiate Estimator
        
    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir=LOGDIR, params=model_params)
    
    # Train
    nn.train(input_fn=train_in, steps=num_iter)
    
    # Score the accuracy
    ev = nn.evaluate(input_fn=test_in) # need to mod the model fn for eval
    print('what is accuracy??:', ev) 
    print('what is accuracy??:', type(ev)) 
    
    ## print out predictions here


main()


































