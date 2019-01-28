from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from urllib import request 

import numpy as np
import tensorflow as tf

# Data sets
iris_training = "iris_training.csv"
iris_training_url = "http://download.tensorflow.org/data/iris_training.csv"

iris_test = "iris_test.csv"
iris_test_url = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # if the training and test sets aren't stored locally, downlaod them.
    if not os.path.exists(iris_training):
        raw = request.urlopen(iris_training_url).read().decode() # bytes if not decoded
        with open(iris_training, 'w') as f:
            f.write(raw)

    if not os.path.exists(iris_test):
        raw = request.urlopen(iris_test_url).read().decode()
        with open(iris_test, 'w') as f:
            f.write(raw)

    
    # loading the datasets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_training, target_dtype=np.int, features_dtype=np.float32)
    test_set =tf.contrib.learn.datasets.base.load_csv_with_header(filename=iris_test, target_dtype=np.int, features_dtype=np.float32)
    ## just defined the tuples containing the datasets for train and test>

    ### modeling NN model
    # specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("X", shape=[4])] # 4 features in X

    # build 3 layer DNN with 10, 20, 10 units respectively. 
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir='/home/wataru/machineLearn/kaggle/TF_practice/estim_dir', activation_fn=tf.nn.sigmoid) # model_dir is the dir where TF stores the record data

    #### describe the training input pipeline
    ''' the tf.estimator API uses input functions, which create the TF operations
    that generates data for the model. We can use tf.estimator.input.numpy_input_fn
    to produce the input pipeline.
    '''

    def my_input_fn(sets):
        print(sets.data)
        print(sets.target)
        return tf.estimator.inputs.numpy_input_fn(x={"X":np.array(sets.data)}, 
                y=np.array(sets.target), num_epochs=None, shuffle=True)
    # Define the training input
    # this will be passed into train method of estimator
    #input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": np.array(training_set.data)}, y=np.array(training_set.target), num_epochs=None, shuffle=True)

    ##### Fit the DNNClassifier to the Iris Training Data
    '''
    New that DNN classifier is configed, we can fit it to the Iris training data using the train method. 
    Pass train_input_fn as the input_fn, and the number of steps to train(here, 200)
    '''
    # train model
    classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)
    '''
    The state of the model is preserved in the classifier, which means you can 
    train iteratively if you like. For example, the above is equivalent to the 
    following.
    '''
    ## classifier.train(input_fn = train_input_fn, steps=1000)
    ## classifier.train(input_fn = train_input_fn, steps=1000)

    '''
    However, if you're looking to track the model while it trains, you'll likely
    want to instead use a Tensorflow SessionRunHook to perform logging ops.
    '''

    ##### Eval the model accuracy #####
    '''
    You've trained your DNNClassifier model on the Iris training data; now you 
    can check its accuracy on the Iris test data using the evaluate method. 
    Like train, evaluate takes an 'input function' that builds its input pipeline.
    evaluate returns a dictionaries with the evaluation results. 
    The following code passes the iris test data, test_set.data and test_set.target
    to evaluate and prints the accuracy from the results:
    '''
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'X':np.array(test_set.data)}, y=np.array(test_set.target), num_epochs=1, shuffle=False)
    '''
    Note: The num_epochs=1 argument to numpy_input_fn is important here.
    test_input_fn will iterate over the data once, and then raise OutOfRangeError.
    This error signals the classifier to stop evaluating, so it'll evaluate over
    the input once.
    '''

    # Eval accuracy
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)['accuracy']

    print('\nTest Accuracy: {0:f}\n'.format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": new_samples},num_epochs=1,shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p['classes'] for p in predictions]

    print('New samples, class predictions:  {}\n'.format(predicted_classes))

if __name__=='__main__':
    main()
























