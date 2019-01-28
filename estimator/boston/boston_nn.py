from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf

#### continue of the Input_fn tutorial on offical TF website###

### A Neural Network Model for Boston House Values #####

'''
In the remainder of this tutorial, you'll write an input function for 
preprocessing a subset of Boston housing data pulled from the UCI Housing Data Set
and use it to feed data to a neural network repgressor for predicting median house 
values.

The Boston CSV data sets you'll use to train contains the following feature data
for Boston suburbs:

    CRIM: crime rate per capita
    ZN: fraction of residential land zoned to permit 25,000+ sq ft lots
    INDUS: fraction of land that is non-retails business
    NOX: concentration of nitric oxides in part per 10 million
    RM: Average Rooms per dwelling
    AGE: fraction of owner-occupied residences built before 1940
    DIS: distance to Boston-area employment centers
    TAX: property tax rate per $10, 000
    PTRATIO: student-teacher ratio

And the label your model will predict is MEDV, the median value of owner-occupied residences in thousands of dollars.
'''

##### Setup #####

'''
Importing the housing data:
    To start, set up your imports (including pandas and TF) and set logging
    verbosity to INFO for more detailed log output:
'''

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

'''
Define the column names for the data set in COLUMNS. To distinguish features from
the label, also define FEATURES and LABEL. Then read the three CSVs(tf.train, tf.test, and predict) into pandas DataFrames:
'''

COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 
        'medv']
# columns is when looking at the whole data, inclujding X and y

FEATURES = ["crim", "zn", "indus", "nox", "rm", 'age', 'dis', 'tax', 'ptratio']
# features only looks at X

LABEL = 'medv'
# labels only y, that's why medv is missing from Features

training_set = pd.read_csv('boston_train.csv', skipinitialspace=True, 
                            skiprows=1, names=COLUMNS)
test_set = pd.read_csv('boston_test.csv', skipinitialspace=True, 
                            skiprows=1, names=COLUMNS)

prediction_set = pd.read_csv('boston_predict.csv', skipinitialspace=True, 
                            skiprows=1, names=COLUMNS)


### Defining FeatureColumns and Creating the Regressor

'''
Next, create a list of FeatureColumns for the input data, which formally specifiy 
the set of features to iuse for training. Because all features in the housing dataset
contain continuous values, you can create their FeatureColumns using the 
tf.contrib.layers.real_valued_column() function:
'''

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

'''
NOTE: For more in-depth overview of feature columns, see https://www.tensorflow.org/tutorials/linear#feature_columns_and_transformations
, and for an example that illastrates how to define FeatureColumns for categorical
data, see the Linear Model Tutorial (same link as above??)

Now, instantiate a DNNRegressor for the neural network regression model. You'll 
need to provide two arguments here: hidden_units, a hyperparameter specifying the
number of nodes in each hidden layer(here, two hidden layers with 10 nodes each), 
and feature_columns, containing the list fo FeatureColumns you just defined:
'''

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10], 
                                      model_dir='/home/wataru/machineLearn/kaggle/TF_practice/estimator')


### Building the input_fn

'''
To pass input data into the regressor, write a factory method that accepts a
pandas DataFrame and returns an input_fn:
'''
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k: data_set[k].values for k in FEATURES}), 
            y = pd.Series(data_set[LABEL].values),
            num_epochs=num_epochs,
            shuffle=shuffle)

    ## {k:dataset[k].values for k in FEATURES} itering through each feature columns and returning numpy array of data for the columns

'''
Note that the input data is passed into input_fn in the data_set argument, which
means the function can process any of the DataFrames you've imported:
    training_set, test_set, and prediction_set.
'''


'''
Two additional arguemnts are provided:
    num_epochs: congtrols the number of epochs to iterate over data. For training,
    set this to None, so the input_fn keeps returning data until the required # of
    train steps is reached. 
    For evaluate and predict, set this to 1, so the input_fn will iterate over the 
    data once and then raise OutOfRangeError. That error will signal tthe Estimator
    to stop evaluate or predict.
    Shuffle: Whether to shuffle the data. For evaluate and precdict, set this to
    False, so the input_fn iterates over the data sequentially. For train, 
    set this to True.
'''

##### Training the Regressor #####

'''
To train the neural network regressor, run train with the training_set passed to
the input_fn as follows:
'''
regressor.train(input_fn=get_input_fn(training_set), steps=5000)

'''
You should see log output similar to the following, which reports training loss
for every 100 steps:
    INFO:tensorflow:Step 1: loss = 483.179
    INFO:tensorflow:Step 101: loss = 81.2072
    INFO:tensorflow:Step 201: loss = 72.4354
    ...
    INFO:tensorflow:Step 1801: loss = 33.4454
    INFO:tensorflow:Step 1901: loss = 32.3397
    INFO:tensorflow:Step 2001: loss = 32.0053
    INFO:tensorflow:Step 4801: loss = 27.2791
    INFO:tensorflow:Step 4901: loss = 27.2251
    INFO:tensorflow:Saving checkpoints for 5000 into /tmp/boston_model/model.ckpt.
    INFO:tensorflow:Loss for final step: 27.1674.
'''

### Evaluating the Model ###

'''
Next, see how the trained model performs against the test dataset. Run evaluate,
and this time pass the test_set to the input_fn:
'''
ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, 
                        shuffle=False))

# Retrive the loss from the ev result and print it to output.

loss_score = ev['loss']
print('Loss: {0:f}'.format(loss_score))

'''
You should see results similar to the following:

INFO:tensorflow:Eval steps [0,1) for training step 5000.
INFO:tensorflow:Saving evaluation summary for 5000 step: loss = 11.9221
Loss: 11.922098
'''
### Making Predictions ###

'''
Finally, you can use the model to predict median house values for the 
prediction_set, which contains feature data but no labels for six examples:
'''
y = regressor.predict(
        input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))

# .predict() returns an iterator of dicts; convert to a list and print predictions
predictions = list(p['predictions'] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
















































