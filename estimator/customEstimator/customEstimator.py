from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Import urllib
from six.moves import urllib
'''
Google's TensorFlow tutorial, creating Estimators in tf.estimator to fit your needs
in case the default estimator won't. 

This tutorial covers how to create your own Estimator using the building blocks
provided in tf.estimator, which will predict the ages of abalones based on their
physical measurements. You'll learn how to do the following:

    Instantiate an Estimator
    Construct a custom model function
    Configure a neural network using tf.feature_column and tf.layers
    Choose an appropriate loss function from tf.losses
    Define a training op for your model
    Generate and return predictions

'''
### An Abalone Age Predictor
# estimate the age of abalone (sea snail) by the number of rings on its shell.

'''
THe abalone dataset contains the feature data for abalone:
    Length, diameter, height, whole weight (weight of whole in grams)
    shuckedc weight, viscera weight, shell weight
'''
'''
Setup

This tutorial uses three data sets. abalone_train.csv contains labeled training data comprising 3,320 examples. abalone_test.csv contains labeled test data for 850 examples. abalone_predict contains 7 examples on which to make predictions.

The following sections walk through writing the Estimator code step by step; the full, final code is available here.
'''

## Loading teh csv data into TF datasets
import numpy as np
import tensorflow as tf
FLAGS = None

## Enable logging:
tf.logging.set_verbosity(tf.logging.INFO) ##???
# logging the tf data

## 

# Then define a function to load the CSVs 
def maybe_donwload(train_data, test_data, predict_data):
    '''Maybe download trainaing data and returns train and test file names'''
    if train_data: # what does this even evaulate??
        train_file_name = train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False) # ??
        urllib.request.urlretrieve(
                "http://download.tensorflow.org/data/abalone_train.csv",
                train_file.name)
        train_file_name = train_file.name #??
        train_file.close()
        print('Training data is downloaded to %s' % train_file_name)


    if test_data: # maybe evaluating the input to the function??
        test_file_name = test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False) # what is this library and method?
        urllib.request.urlretrieve(
                "http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
        test_file_name = test_file.name #?? test_file is class and .name is method??
        test_file.close()
        print('Test data is downloaded to %s' % test_file_name)

    if predict_data:
        predict_file_name = predict_data
    else:
        predict_file = tempfile.NameTemporaryFile(delete=False)
        urllib.request.urlretrieve(
                "http://download.tensorflow.org/data/abalone_predict.csv",
                predict_file.name)
        predict_file_name = predict_file.name
        predict_file.close()
        print("Prediction data is downloaded to %s" % predict_file_name)


    return train_file_name, test_file_name, predict_file_name

''' Finally, create main() and load the abalone CSCs into Datasets, definng flags to
allow users to optionally specify CSV files for training, test, and prediction 
datasets via the command line.
'''

def main(unused_argv):
    # Load datasets
    abalone_train, abalone_test, abalone_predict = maybe_download(
            FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

    # Training examples 
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)
    # load dataset from csv file without the header row as tf....dataset obj

    # Test examples 
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

    # set of 7 examples for which to predict abalone ages
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    #Set model params
    model_params = {'learning_rate': LEARNING_RATE}
    # instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() ## what the heck is parser??
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
            '--train_data', type=str, default='', help='Path to the training data')
    parser.add_argument(
            '--test_data', type=str, default='', help='path to the test data')
    parser.add_argument(
            '--predict_data', type=str, default='', help='Path to the prediction data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) ## what's tf.app.run??


## Instantiating an Estimator

'''
When you're creating your own estimator from scratch, the constructor accepts just
two high-level parameters for model configuration, model_fn and params:
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    model_fn: a function object that contains all teh aforementioned logic to
    support training, evaluation, and prediction. You are responsible for
    implementing that functionality. The next section, constructing the model_Fn
    covers creating a model funciton in detail.

    params: An optional dict of hyperparameters (eg, learning rate, dropout) that
    will be passed into the model_fn

    NOTE: just liek the tf.estimator's predifined regressors an dclassifiers, 
    the Estimator iniotializer also accepts the general configuration arguments
    model_dir and config
'''
'''
For the abalone age predictor, the model will accept one hyperparameter: 
learning rate. Define LEARNING_RATE as a constant at the beginning of your code,
right after the loggin configuration:
    tf.logging.set_verbosity(tf.logging INFO)

    ## Learning rate
    LEARNING_RATE = 0.001


Then add the following code to main(), which creates the dictionary model_params 
containing the learning rate and instantiates the Estimator.
    # set model params
    model_params = {'learning_rate': LEARNING_RATE}
    # instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params = model_params)
'''

## Constructing the model_fn
#The basic skeleton for an Estimator API model function looks like this:
def model_fn(features, labels, mode, params)

    # Logic to do the following:
    # 1. configure the model via Tensorflow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)


'''
The model_fn must accept three argments:
    features: A dict containing the features passed to the model via input_fn.
    labels: A tensor containing the labels passed to the model via input_fn. Will
    be empty for predict() calls, as these are the values the model will infer.

    mode: one of the following tf.estimator.Modekeys string values indicating the 
    context in which the model_fn was invoked:
        
        tf.estimator.Modekeys.TRAIN: the model_fn was invoked in training mode, 
        namely via train() call

        tf.estimator.ModelKeys.EVAL: the model_fn was invoked in evaluation mode, 
        namely via na evaluate() call

        tf.estimator.ModeKey.PREDICT: the model_fn was invoked in predict mode, 
        namely via a predict() call.

    model_fn may also accept a params argument containing a dic of hyperparameters
    used for training (as shown in the skelton above)

    THe body of the function performs the following tasks (described in details in the sections below):

    configuring the model-here, for the abalone predictor, this will be a neural network.

    Defining the loss function used to calculate how closely the model's predictions 
    match the target values.

    Defining the training operation that specifies the optimizer algorithm to 
    minimize the loss values calculated by the losss function.

The model_fn must return a tf.estimator.EstimatorSpec object, which contains the follwing values:

    mode: (required) THe mode in which the model was run. Typically, you will return 
    the mode argument of the model_fn here.

    predictions: (required in PREDICT mode) A dict that maps key names of youri choice 
    to Tensor's containing the predictions from model, eg.
    python predictions = {"results": tensor_of_predictions}
    In PREDICT mode, the dict that you return in EstimatorSpec will then be 
    returned by predict(), so you can construct it in the format in which you'd 
    like to consume it.

    loss(required in EVAL and TRAIN mode): A tensor containing a scalar loss value: 
    the output of the model's loss function(cost function) calculated over all
     the input examples. This is used in TRAIN mode for error handling and logging, 
     and is automatically included as a metric in EVAL mode.

     train_opo(required only in TRAIN mode): An Op that runs ONE STEP of training.

     eval_metric_ops(optional): A dict of name/value pairs specifiying the metrics 
     that will be calculated when the model runs in EVAL mode. The name is a label
     of your choice for the metric, and the value is the result of your metric 
     calculation. The tf.metrics module provides predefined functions for a variety
     of common metrics. The following eval_metric_ops contains an 'accuracy' 
     metric calculated using tf.metrics.accuracy:

     python eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels, predictions)}

     If you do not specify eval_metric_ops, only loss will be calculated during
     evaluation.

'''

### Configuring a Neural Network with tf.feature_column and tf.layers #####

''' Constructing a neural netwokr entails creating and connecting the input 
layer, the hidden layers, and the output layer.

The input layer is a series of nodes (one for each feature in the mdoel) that will 
accept the feature data that is passed to the model_fn in the feature argument.
If features contains an n-dimensional Tensor with all your feature data, then it
can serve as the input layer. If features contains a dict of feature columns 
passed to the model via an input function, you can convert it to an input-layer 
Tensor with the tf.feature_column.input_layer function.

input_layer = tf.feature_column.input_layer(feature=features, feature_columns=[
                                age, height, weight])


As shown above, input_layer() takes two required arguments:

    features: A mapping from string keys to the Tensors containing the corresponding
    feature data. This is exactly what passed to teh model_fn in the features 
    argument.

    feture_columns: A list of all the FeatureCoulumns in the model- age, height,
     and weight in the above example.

The input layer of the neural network then must be connected to one or more hidden
layers via an activation function that perfoerms a onlilner transformawtion on the 
data from the previous layer. The hidden layer is then connected to the outpuot 
layer, the final layer in the model. tf.layers provides the tf.layers.dense
function for constructting fully connected layers. The activation is controlled by
the activation argument. Some options to pass to the actuvation arguments are:

    tf.nn.relu: the following code creates a layer of units nodes fully connected 
    to the previous layer input_layer with ReLU actuvation function(tf.nn.relu):
    
    python hidden_layer = tf.layers.dense(inputs=input_layer, units=10, activation=tf.nn.relu)

    tf.relu6: The following codes creates a layer of units nodes fully connected 
    to the previous layer hidden_layer with a ReLU 6 activation function(tf.nn.relu6):

    python second_hidden_layer = tf.layers.dense(inputs=hidden_layer, units=10, activation=tf.nn.relu) # meaning relu6?? 
    
    ## what the heck is relu6 anyways??

    None: The following code creates a layer of units nodes fully connected to
    the previous layer second_hidden_layer with no activation function, just a 
    lineawr transformation:

    python output_layer = tf.layers.dense(
    inputs=second_hidden_Layer, units=3, activation=None)

'''

'''
Other activation functions are possible, eg:'''
output_layer = tf.layers.dense(inputs=second_hidden_layer, units=10, 
        activation=tf.sigmoid)

'''THe above codde creates the neural network layer output_layer, which is fully 
connected to second_hidden_layer with a sigmoid activation function(tf.sigmoid).
For a list of predefined activation functions available in TF, see the API docs.
API: https://www.tensorflow.org/api_guides/python/nn#activation_functions 
'''

'''
Putting it all together, the following code constructs a full neural network for
the abalone predictor, and captures its predictions:
'''

def model_fn(features, labels, mode, params):
    '''Model function for Estimator'''

    ## connect the first hidden layer to input layer
    # (features['x']) with relu activation
    first_hidden_layer = tf.layer.dense(features['x'], 10, activation=tf.nn.relu)

    ## connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, 
            activation = tf.nn.relu)

    # connect the output layer to the second hidden layer (no activation fn)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    # reshape output laeyr to 1D Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1]) # how just -1??
    prediction_dict = {'ages': predictions}


'''
Here, because you'll be passing the abalone Datasets using numpy_input_fn as 
shown below, features is a dict {'x': data_tensor}, so features['X'] is the
input layer. (or you mean features['x'] is the input data as opposed to the label??)

The network contains two hidden layers, each with 10 nodes and a ReLU activation
function. The output layer contains no activation function, and is tf.reshape to
a 1D tensor to capture the model's predictions, whichc are stored in predictions_dict.
## I'm not sure what is the label/ predictions are here, is it the probabilitis
like the usual classification neural network? or is it more like a number like 
regression? how can NN produce a number though? unless... oh the last layer(output
) don't have the activagtion function... that means it'll return the original 
sum and multipleid value, not the one between 0-1.
'''

# Defining loss for the model

'''
The estimatorSpec returned by the model_fn must contain loss: a Tensor representing
the loss value, which quantifies how well the model's predictions reflect the 
label values during training and evaluation runs. The tf.losses module provides
convenience functions for calculating loss using a variety of metrics, including:

    absolute_difference(labels, predictions). Calculate loss using the 
    'absolute-difference formula', also known as L1 loss. 

    los_loss(labels, predictions). Calculates loss using the 
    'logistic loss formula' (typically used in logistic regression)

    mean_squared_error(labels, predictions) calculates loss using the 
    'mean squared error' (MSE, also known as L2 loss)
'''
'''
The following example adds a definition for loss to the abalone model_fn using 
mean_squared_error() 
'''

def model_fn(features, labels, mode, params):
    '''model function for estimator'''

    ## connect the first hidden layer to input layer
    # (feature['x']) with relu activagtion
    first_hidden_layer = tf.layers.dense(features['x'], 10, activation=tf.nn.relu)

    # connect the second hidden layer to first hidden layer with relu
    second_hidden_laeyr = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)

    # connect the output layer to the second hidden layer without activaton fn
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    ## Reshape the output layer to 1D Tensor to return predictions
    ''' so what shape is the original output layer??'''
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {'ages': predictions}

    #### Calculate the LOSS using MSE (MEAN SQUARED ERROR)
    loss = tf.losses.mean_squared_error(labels, predictions)

''' see the API guide for the full list of loss functions and more details on
supported arguments and usage.'''
'''
Supplementaryu metrics for evaluation can be added to an eval_metric_opos dict.
The folowing code defines an rmse metric, which calculates the root mean squared 
eror for the model predictions. Note that the labels tensor is cast to a float64
type to match the data type of the predictions tensor, which will contain real values:
    '''
    eval_metric_ops = {
            'rmse': tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), predictions)
            }

### Defining the training op for the model

'''
The training op defines the optimization algorithmn Tensorflow will use when fitting
the model to the training data. Typically when training, the goal is to minimize
loss. A simple way to create the training op is to instantiate a tf.train.Optimizer
subclass and call the minimize method.

The following code defines a training opp for the abalone model_fn using the loss
value calculated in Defining Loss for the Model, the learning rate passed to the 
function in prarams, and the gradient descent optimizer. For global_step, the 
convenience function tf.train.get_global_step takes ccare of generating an integer
variable:
    '''

    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params['learning_rate']) # param is dictionary of param
    train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())


'''FOr full list of optimizers, and other details, see the API guide. 
https://www.tensorflow.org/api_guides/python/train#optimizers
'''

##### THe Complete abalone model_fn #####
'''
Here's the final, complete model_fn for the abalone age predictor. The following
code condigures the neural network; defines loss and training op; and returns a
EstimatorSpec object containing mode, predictions_dict, loss, and train_op;
'''

def model_fn(features, labels, mode, params):
    '''model function for Estimator.'''

    # connect the first hidden layer to input layuer
    #(feature['x']) with relu activation
    first_hidden_layer = tf.layers.dense(feature['x'], 10, activation=tf.nn.relu)

    # connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)

    # Connect the output layer to the second hidden layer (no activaton function)
    output_layer = tf.layer.dense(second_hidden_layer, 1)

    ## reshape output layer to 1D tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    ## provide an estimator spec for 'ModeKey.PREDICT'
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode, predictions=('ages':predictions))

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)

    # calculate root mean squared error as additional eval metric
    eval_metric_ops = {
            'rmse': tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), predictions)}


    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params['leaning_rate'])
    train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step()) # tf.train.get_global_step() ??? 

    # Provide an estimator spec for 'ModeKey.EVAL' and 'ModeKeys.TRAIN' modes.

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)


##### Running the Abalone Model #####
'''
You've instantiated an Estimator for the abalone predictor and defined its 
behavior in model_fn; all that's left to do is train, evaluate, and make 
predictions.
'''

'''And the following code to the end of main() to fit the neural network to the 
training data and evaluate accuracy:
    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':np.array(training_set.data)}, 
            y = np.array(training_set.target),
            num_epochs=None,
            shuffle=True)


# Train
nn.train(input_fn=train_input_fn, steps=5000)

print('Loss: %s' %ev['loss'])
print('Ross Mean Squared Error: %s' %ev['rmse'])


'''Note: the above code uses input functions to feed feature(x) and label(y)
Tensors into the model for both training(train_input_fn) and evaluation (test_input_fn). 

'''
### To predict ages for the Abalone_predict dataset, add the following line to the main()

# Print out predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': prediction_set.data}, 
        num_epochs=1,
        shuffle=1, 
        shuffle=False)

predictions = nn.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
    print('Prediction %s: %s' %(i +1, p['ages']))

'''Here, the predict() function returns results in predictions as an iterable.
The for loop enumerates and prints out the results. 

For additional References:

    Layers: https://www.tensorflow.org/api_guides/python/contrib.layers 
    Losses: https://www.tensorflow.org/api_guides/python/contrib.losses 
    Optimize: https://www.tensorflow.org/api_guides/python/contrib.layers#optimization( 












































