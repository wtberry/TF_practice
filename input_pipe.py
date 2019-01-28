import numpy as np
import tensorflow as tf

########### Importing Data ##########################
'''
This goes over creating TF input pipleline using tf.conrtib.data.Dataset API
'''
## For now only about taking data from numpy arrays, and processing it with the API, 
## so it can be fed into feeding dictionary in TF sessions. 
## This should also support batching of the data as well as random shuffling ###

## Official website:
#  www.tensorflow.org/programmers_guide/datasets  

'''
The Dataset API enables you to build complex input pipelines from simple, reusable pieces.
For example, the pipeline for an image model might aggregate data from files in a 
distributed file system, apply random perturbations to each image, and merge randomly 
selected images into a batch for training. The pipeline for a text model might involve 
extracting symbols from raw text data, converting them to embedding identifiers with a 
lookup table, and batching together sequences of different lengths. 
'''

### the Dataset API introduces two new abstractions to Tensorflow:
'''
A tf.contrib.data.Dataset - represents a sequence of elements, in which each element 
    contains one or more Tensor objects. e.g, in an image pipeline, an element might be a 
    single training example, with a pair of tensors representing the image data and a 
    label. There are two distinct ways to create a dataset:

        1. Create a source(e.g. Dataset.from_tensor_slices()) constructs a dataset from 
        one or more tf.Tensor objects.

        2. Applying a transformation(e.g. Dataset.batch()) constructs a dataset from 
        one or more tf.contrib.data.Dataset objects.

        3. A tf.congtrib.data.Iterator provides the main way to extract elements from 
        Dataset. The operation returned by Iterator.get_next() yields the next element of 
        a Dataset when executed, and typically acts as the interface between input 
        pipeline code and your model. The simplest iterator is a "one-shot-iternator", 
        is associated with a particular Dataset and iterates through it once. 
        For more sophisticated uses, the Iterator.initializer operation enables you to 
        reinitialize and parameterize and iterator with different datasets, so that you 
        can, for example, iterate over training and validation data mutiple times in the 
        same program.
'''
### Basic Mechanics ###
'''
To start an input pipelkne, you must define a source. e.g.
    to construct a Dataet some tensors in memory, use
    tf.contrib.data.Dataset.from_tensor() or
    tf.contrib.data.Dataset.from_tensor_slices(). 
    ## you can pass in numpy matrics to those arguments by using
    tf.convert_to_tensor(np_matrix)

    Alternatively, if your input data are 
    on disk in hte recommended TFRecord format, can construt a 
    tf.contrib.data.TFRecordDataset

Once you have the Dataset object, you can apply pre-mutation on the data, such as 
    Dataset.map(), Dataset.batch()

    for more info: if you would like t://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset  



    After that, you can get each (1 data) from the dataset by iterating through it, by
    creating the iterator object. 
    THe iterator obj provides 2 options:
        1. Iterator.initializer: enables you to (re)initialize the iterator's state, and
        2. Iterator.get_next(): which returns tf.Tensor obj that corresponds to the 
        symbolic next element.

        Depending on cases, different kinds of iterators are available. Will be covered
        next...
'''

##### Dataset Structure #####
'''
Nested structures of ...
A dataset is composed of...
    elements(same structures)
        Components: one or more tf.Tensor obj 
            tf.DType - represent the tuype of elements in the tensor
            tf.TensorShape - represents shape (static or partially decided) of element

## Dataset.output_types and Dataset.output_shapes enables to inspect the types and shapes
of each component of a dataset element.
Those structures can be nested dictionaries and stuff, refer to the official website for 
more examples

When dealing with these structures, it's convenient to name each component of an element.
e.g     if they represent different features of training example. In addition to tuple,
        collections.namedtuple or a dictionary mapping strings to tensors are available
'''
## eg:
dataset = tf.contrib.data.Dataset.from_tensor_slices(
        {'a': tf.random_uniform([4]),
            'b': tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})

print(dataset.output_types)
print(dataset.output_shapes)

'''
THe dataset transformations support datasets of any structure. When using the 
Dataset.map(), Dataset.flat_map(), and Dataset.filter() transformations, which apply a 
function to each element, the elemnet structure determines the argumnet of the function.
'''

dataset1 = dataset1.map(lambda x:...)
dataset2 = dataset2.flat_map(lambda x, y :...)

# Note: Argumnet destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)

## Creating an iterator ###
'''
Once you have built a Dataset to represent your input data, the next step is to create an 
Iterator to access elements from that dataset. THe Dataset API currently supports three
kinds of iterator, in increasing level of sophistication:

    one-shot,
    initializable,
    reinitializable, and
    feedable


A one-shot iterator is the simplest form of iterator, which only supports iterating once 
through a dataset, with no need for explicit initialization. One-shot iterators handle
almost all of the cases that the existing queue-based input pipeline support, but they 
do not support parameterization. Using the example of Dataset.range():
'''
dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value


'''
An initializable iterator requires you to run an explicit iterator.initializer operation
before using it. It enables you to parameterize the definition of the dataset, using one
or more tf.placeholder() tensors that can be fed when you initialize the iterator. 
Continuing the Dataset.range() example:
'''
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.contrib.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

## Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
## feed dictionary = feed_dict={<placeholder's_name>: value}
for i in range(10):
    value = sess.run(next_element)
    assert i == value

## initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
    value = sess.run(next_element)
    assert i == value

'''A reinitializable iterator can be initialized from multiple different Dataset objs.
For example, you might have a training input pipeline that uses random perturbations to
input images to improve generalization, and validation input pipeline that evaluagtes 
predictions on unmodified data. These pipelines will typically use different Dataset objs
that have the same structure(i.e. the same types and compatible shapes for each component)
'''
# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the 'output_types',
#'output_shapes' properties of either 'training_dataset' or 'validation_dataset' here,
# because they are compatible.
iterator = Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

## Run 20 epochs in which teh training dataset is traversed, followed by validation
## validation dataset.
for _ in range(20):
    # initialize an iterator over the training dataset.
    sess.run(training_init_op)
    for _ in range(100):
        sess.run(next_element)

# Initialize an iterator over the validation set.
sess.run(validation_init_op)
for _ in range(50):
    sess.run(next_element)


'''
A Feedable iterator can be used togeter with tf.placeholder to select what Iterator to 
use in each call to tf.Session.run, via the familiar feed_dict mechanism. It offers the 
same functionality as a reinitializable iterator, but it does not require you to 
initialize the iterator from the start of a dataset when you switch between iterators.
e.g., using the same training and validation example from above, you can use 
tf.contrib.data.Iterator.from_string_handle to define a feedable iterator that allows you
to switch between the two datasets:
'''
# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
        lambda x:x+tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.contrib.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We could use
# the 'output_types' and 'output_shapes' properties of either 'training_dataset' or
# 'validation_dataset' here, because they have identical structure

handle = tf.placeholder(tf.string, shape[]) # why string??
iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with avariety of different kinds of iterator
# (such as one-shot and initializable iteraotors).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The 'Iterator.string_handle()' method returns a tensor that can be evaluated
# and used to feed the 'handle' placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# loop forever, alternating between training and validation.
while True:
    # run 200 steps using the training set. Note that the training set is infinite, and we
    # resume from where we left off in the previous 'while' loop iteration.
    for _ in range(200):
        sess.run(next_element, feed_dict={handle: training_handle})

    # run one pass over the validation set.
    sess.run(validation_iterator.initializer) # have to initialize the iterator
    for _ in range(50):
        sess.run(next_element, feed_dict={handle: validation_handle})

### Consming values from an iterator ####
'''
THe Iterator.get_next() method returns one or more tf.Tensor objects that corresponds to
the symbolic next element of an iterator. Each time these tensors are evaluated, they take
the value of the next elemnet in the underlying dataset. (Note that, like other stateful
objects in Tensorflow, calling Iterator.get_next() does not immediately advance the
iterator. Instead you must use the returned tf.Tensor objects in a TensorFlow expression,
and pass the result of that expression to tf.Session.run() to get the next elemnet and 
advance the iterator.)

If the iterator reaches the end of the dataset, executing the Iterator.get_next() 
operation will raise a tf.errors.OutOfRangeError. Afeter this point the iterator will be
in an unusable state, and you must initialize it again if you want to use it further.
'''
dataset = tf.contrib.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically 'result' will be the output of a model, or an optimizer's training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))
print(sess.run(result))

try:
    sess.run(result)
except tf.errors.OutOfRangeError:
    print('ENd of dataset') 

## A common pattern is to wrap the 'training loop ' in a try-except block:
sess.run(iterator.initializer)
while True:
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        break

# If each element of the dataset has a nested structure, the return value of 
# Iterator.get_next() will be one or more tf.Tensor objects in the same nested structure

dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform[4, 10])
dataset2 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4]), tf.random_uniform([4, 100]))
dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()

'''
Note that evaluating any of next1, next2 or next3 will advance the iterator for all 
components. A typical consumer of an iterator will include all components in a single
expression.
'''

## Reading input data

# Consuming Numpy arrays
'''
If all of your input data fit in memory, the simplest way to create a Dataset from them
is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices()
'''
# Load the training data into two Numpy arrays, for example using 'np.load()'
with np.load('/path/to/your/data/in/numpy.npy') as data:
    features = data['features']
    labels = data['labels']


# assume that each row of 'features ' corresponds to the same row as 'labels'
assert features.shape[0] == labels.shape[0]


dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))

##### DO THIS #### MAKE DATASET WITH PLACEHOLDER AND FEED NUMPY DATA INTO IT
'''
Note that above code snippet will embed the features and labels arrays in your TensorFlow
graph as tf.constant() operations. This works well for a small dataset, but wastes memory-
because the contents of the array will be copied multiple times-and can run into 2GB limit
for the tf.GraphDef protocol buffer.

As an alternative, you can define the Dataset in terms of tf.placeholder() tensors, and
FEED the NumPy arrays when you initialize an Iterator over the dataset.
'''
# Load the training data into 2 numpy arrays, e.g. using 'np.load()'
with np.load('path to your data/file.npy') as data:
    features = data['features']
    labels = data['labels']

# Assume that each row of 'features' corresponds to the same row as 'labels'
assert features.shape[0] == labels.shape[0] # comparing the # of data examples

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
## remeber how to define placeholder??
# tf.placeholder(dtype, shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on 'dataset']

dataset = ....

iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
    
### Consuming TFRecord data ###

'''
THe dataset API supports a variety of file formats so that you can process large datasets
that do not fit in memory. E.g., the TFRecord file format is a simple record-oriented 
binary format that many TensorFlow applications use for training data. The 
tf.contrib.data.TFrecordDataset class enables you to stream over the contents of one or 
more TFRecord files as part of an input pipeline.
'''
# Create a dataset that reads all of the examples from two files.
filenames = ['/path/to/your/file.tfrecord', '/another/path/to/other/file.tfrecord']
dataset = tf.contrib.data.TFRecordDataset(filenames)

'''
The filenames argument to the TFRecordDataset initializer can either be a string, a list 
of string, or a tf.Tensor of strings. Therefore if you have two sets of files for training
and validation purposes, you can use a tf.placeholder(tf.string) to represent the 
filenames, and initialize an iterator from the appropriate filenames:
'''

filenames = tf.placeholder(tf.string, shape=[None]) # define a placeholder for TFR
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map() # Parse the record into tensors
dataset = dataset.repeat() # Repeat the input indefinitely.
dataset = dataset.batch(32) 
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current phase of 
# execution, e.g. training vs validation.

# initialize 'iterator' with training data.
training_filenames = ['/paht/to/training.tfrecord', '/path/to/file2.tfrecord']
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

##### Consuming Text Data #####
'''
Many dataset are distributed as one or more text files. The tf.contrib.data.TextLineDataset
 provides an easy way to extract lines from one or more text files. Given one or more 
 filenames, a TextLineDataset will produce one string-valued element per line of those 
 files. Like a TFRecordDataset, TExtLineDataset accepts filenames as a tf.Tensor, so you
 can parameterize it by passinga tf.placeholder(tf.string)
 '''

filenames = ['/path/to/the/textfile.txt', '/path/to/another/textfile.txt']
dataset = tf.contrib.data.TextLineDataset(filenames)

'''
By default, TextLineDataset yields every line of each file, which may not be desirable, 
e.g. if the file starts with a header line, or contains comments. These lines can be 
removed using the Dataset.skip() and Dataset.filter() transformations.
To apply these traknsformations to each file separatelyu, we use Dataset.flat_map() to 
create a nested Dataset for each file.
'''
filenames = ['/one/text/data/file.txt', '/another/text/data/file.txt']

dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)

# Use 'Dataset.flat_map()' to transform each file as a separate nested dataset.
# and then concatenate their contents sequentially into a single 'flat' dataset.
# * Skip the first line(header row).
# * Filter out lines beginning with '#' (comments).
dataset = dataset.flat_map(
        lambda filename: (
            tf.contrib.data.TextLineDataset(filename).skip(1).filter(
                lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))))


##### Preprocessing data with Dataset.map() #####
'''
The dataset.map(f) transformation produces a new dataset by applying a given function f
to each element of the input dataset. It is based on the map() function that is commonly
applied to lists(and other structures) in functional programming languages. 
The function f takes the tf.Tensor objs that represent a single element in the input, and 
 returns tf.Tensor objs that will represent a single element in the new dataset. 
 Its implementation uses standard TensorFlow operations to transform one element into
 another. 
 
 This section covers common examples of how to use Dataset.map()
 '''

 ### few parts skipped ##
 # parsing tf.Example protocol buffere messages
 # Decording image data and resizing it
 # Applying arbitray Python logic with tf.py_func()
 # 

 ##### Batching dataset elements #####
# simple batching 
'''
The simplest form of batching stacks n consecutive elements of a dataset into a single
element. The dataset.batch() transformation does exactly this, with the same constraints
as the tf.stack() operator, applied to each component of the elements: i.e. for each 
component i, all elements must have a tensor of the exact same shape.
'''

inc_dataset = tf.contrib.data.Dataset.range(100)
dec_dataset = tf.contrib.data.Dataset.range(0, -100, -1)
dataset = tf.contrib.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))

### Batching Tensors with Padding ###
'''
The above recipe works for tensor that all have the same size. However, many models(e.g 
sequence models) work with input data that can have varying size (e.g. sequences of 
differnt lengths). To handle this case, the Dataset.padded_batch() transformation enables
you to batch tensors of different shape by specifying one or more dimensions in wwhich 
they may be padded.
'''
dataset = tf.contrib.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x)) # tf.fill?
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))
print(sess.run(next_element))

'''
The Dataset.padded_batch() trnasformation allows you to set different padding for each
dimension of each component, and it may be variable-length (signified by NOne in the 
example above) or constant-length. It is also possible to override the padding value, 
which defaults to 0
'''

### Training Workflows ###

# Processing multiple epochs
'''
The Dataset API offers to main ways to process multiple epochs of the same data.

The simplest way to iteratte over a dataset in multiple epochs is to use the 
Dataset.repeat() transformation. E.g., to create a dataset that repeats its input for
10 epochs:
'''
filenames = ['/path/to/file1.tfrecord', '/var/path/to/file2.tfrecord']
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)

'''
Appying the Dataset.repeat() transformatino with no arguments will repeat the input 
indefinitely. The Dataset.repeat() transformation concatenates its arguments without 
signaling the end of one epoch and the beginning of the next epoch.

if you wish to receive a signal at the end of each epoch, you can write a trainng loop
that catches the tf.errors.OutOfRangeError at the end of a dataset. At that point you
might collect some statistics(e.g. the validation error) for the epoch.
'''
filenames = ['/var/path/to/file.tfrecord', '/another/record/file.tfrecord']
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map()
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element - iterator.get_next()

# compute for 100 epochs,
for _ in range(100):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        # perform end of epoch calculations here.

        # tf.contrib.data.Dataset.from_tensor_slices(tuple of feature and labels, numpy)

### Randomly shuffling input data ###
'''
The Dataset.shuffle() transformation randomly shuffle the input dataset using a similar
algorithm to tf.RandomShuffleQueue: it maintains a fixed-size buffer and chooses the next
element uniformly at random from that buffer.
'''
filenames = ['/poath/to/the/file.tfrecord', ....]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map()
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset=  dataset.repeat()

### Using high-level APIs ###
'''
THe tf.train.MonitoredTrainingSession API simplifies many aspects of running TEnsorFLow in
a distributed setting. MonitoredTrainingSession uses the tf.errors.OutOfRangeError to 
signal that training has completed, so to use it with the Dataset.make_one_shot_iterator().
For example:
'''
filenames = ['/var/path/to/tfrecord']
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)

training_op = tf.train.AdagradOptimizer(...).minizie(loss)

with tf.train.MonitoredTrainingSession(...) as sess:
    while not sess.should_stop():
        sess.run(training_op)


'''
To use a Dataset in the input_fn of a tf.estimator.Estimator, we also recommend
using Dataset.make_one_shot_iterator(). For example:
'''
def dataset_input_fn():
    filenames = ['/var/path/to/file.tfrecord']
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # use 'tf.parse_single_example()' to extract data from 'tf.Example'
    # protocol buffer, and perform any additional per-record preprocessing.

    def perser(record):
        keys_to_features={
                'image_data':tf.FixedLenFeature((), tf.string, default_value=""),
                'date_time': tf.FixedLenFeature((), tf.int64, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=
                    tf.zeros([], dtype=tf.int64)),
                }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.decode_jpeg(parsed['image_data'])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed['label'], tf.int32)
    
        return {'image_data': image, 'date_time': parsed['date_time']}, label
    
    # Use 'dataset.map()' to build a pair of a feature dictionary and a label tensor
    # for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    # 'features' is a dictionry in which each value is a batch of values for that 
    # feature; 'labels' is a batch of labels.
    
    features, labels = iterator.get_next()
    return features, labels
    















































