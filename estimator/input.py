import tensorflow as tf
import numpy as np

### custom input pipelines with input_fn

'''
The input_fn is used to pass feature and target data to the train, evaluate, and predict methods of Estimator. 
The user can do feature engineernig or pre-processing inside the input_fn.
'''
## anatomy of input_fn
# the following code illustrates the basic skeleton for an input function:

def my_input_fn():
    # preprocess your data here

    # .. then return 1) a mapping of feature columns to Tensor with 
    # the corresponding feature data, and 2) a Tensor containing labels 
    return feature_cols, labels

'''
THe body of the input fun contains the specific logic for preprocessing your input
data, such as scrubbing out bad examples or feature scaling.

input functions must return the following values containing the final feature
and label data to be fed into your model(as shown in the above code skelton):
'''

'''
feature_cols
    A dict containing key/value pairs that map feature column names to Tensor s
    (or SparseTensor s) containing the corresponding feature data.

labels
    A Tensor containing your label(target) values: the values your model aims to 
    predict.
'''

#### Converting Feature Data to Tensors

'''
If your feature/label data is a python array or stored in pandas dataframes or numpy arrays, you can use the following methods to construct input_fn:
'''

# numpy input_fn
my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x':np.array(x_data)}, 
        y=np.array(y_data),
        ...)

# pandas input_fn
my_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({'x': x_data}),
        y=pd.Series(y_data),
        ...)

'''
For sparse, categorical data(data where the majority of values are 0), you'll 
instead want to populate a SparseTensor, which is instantiated with three arguments:
    dense_shape
        THe shape of the tensor. Takes a list indicating the number of elements 
        in each dimension. For example, dense_shape=[3,6] specifies a two-dimensional
        3x6 tensor, dense_shape=[2,3,4] specifies a three-demensional 2x3x4 tensor,
        and dense_shape=[9] specifies a one-dimensional tensor with 9 elements.

    indices
        The indices of the elements in your tensor that contain nonzero values.
        Takes a list of terms, where each term is itself a list containing itself 
        a list containing the index of a nonzero element.
        (Elements are zero-indexed-i.e., [0,0] is the index value for the element
        in the first column of the first row in a 2D tensor.) For ex, 
        indices=[[1,3], [2,4]] specifies that the elements with indexes of 
        [1,3] and [2,4] have non zero values.

    values
        A 1D tensor of values. Term i in values corresponds to term i in indices and specifies its value.
        For ex, given indices=[[1,3], [2,4]] the parameter values=[18, 3.6] specifies that element [1,3] of the tensor has a value of 18, and element [2,4] of the 
        tensor has a value of 3.6.
'''

'''
The following code defines a 2D SparseTensor with 3 rows and 5 columns. 
The element with index[0,1] has a value of 6, and the element with index [2,4] has
a value of 0.5. (all other values are 0)
'''

sparse_tensor=tf.SparseTensor(indices=[[0,1], [2,4]], values=[6, 0.5], dense_shape=[3, 5])

#### Passing input_fn Data to your model
'''
To feed data to your model for training, you simply pass the input function you've 
created to your train operation as the value of the input_fn parameter, eg..
'''
classifier.train(input_fn=my_input_fn, steps=2000)

'''
Note that the input_fn parameter must receive a function object(ie, input=my_input_fn), 
not the return value of a function call (input_fn=my_input_fn()). 
This means that if you try to pass parameters to the input_fn in your train call,
as in the following code, it will result in a TypeError:
'''
classifier.train(input_fn=my_input_fn(training_set), steps=2000)

'''
Howerver, if you'd like to be able to parameterize your input function, there are 
other methods for doing so. You can employ a wrapper function that takes no 
argument as your input_fn and use ti to invoke your input function with the
desired parameters. For example...
'''

def my_input_fn(data_set):
    ...

def my_input_fn_training_set(): ## 'wrappin' the value of my_input_fn in a func
    return my_input_fn(training_set)

classifier.train(inpu_fn=my_input_fn_training_set, steps=2000)

'''
Alternatively, you can use Python's functools.partial function to construct a new 
function object with all parameter values fixed:
'''
classifier.train(
        input_fn=functools.partial(my_input_fn, data_set=training_set),
        steps=2000)

'''
The third option is to wrap your input_fn invocation in a lambda and pass it to 
the input_fn parameter:
'''
classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)

'''
One big advantage of designing your input pipeline as shown above-to accept a 
parameter for data set- is that you can pass the same input_fn to evaluate and
predict operations by just changing the dataset argumenmt, e.g:
'''
classifier.evaluate(input_fn=lambda: my_input_fn(test_set), steps=2000)

'''
This approach enhances code maintainability: no need to define multiple input_fn
(e.g input_fn_train, input_fn_test, input_fn_predict) for each type of operation.
'''

'''
Finally, you can use the methods in tf.estimator.inputs to create input_fn from 
numpy or pandas data sets. The additional benefit is that you can use more arguments,
such as num_epochs and shuffle to congtrol how the input_fn iterates over the data:
'''

def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame(...),
            y=pd.Series(...), 
            num_epochs=num_epochs,
            shuffle=shuffle)

def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
            x={...},
            y=np.array(...),
            num_epochs=num_epochs,
            shuffle=shuffle)

## neural nets input using the techunique, continue to 
## boston_nn.py
































