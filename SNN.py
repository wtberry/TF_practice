## practice of building Simple Neural Network using TensorFlow, applied on 
## the MNIST dataset

import tensorflow as tf
## loading the data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### network creation ###
numClasses = 10 # #of classification labels
inputSize = 784 # size of input layer, 28^2
numHiddenUnits = 50 # size of hidden units
batchSize = 100 # batch size to feed into training 
trainingIterations = 10000 # training iteration loop

## defining place holders for input and outputs
# we can define placeholder, and put certain values/sizes later on
tf.reset_default_graph() # make sure that we start with new graph

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape = [None, inputSize], name='x-input') #None means that can be added after, batchSize 
    y = tf.placeholder(tf.float32, shape=[None, numClasses], name='y-input') # #ofExample x labels
    
### Defining weights and bias terms
# Tips: 
'''The key when defining these variable is to keep track of the shapes and to make sure that the output matrix of the previous layer is able to multiply with the weight matrix of the next layer.'''
with tf.name_scope('weights'):
    w1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numClasses], stddev=0.1))
    ## tf.Variable is updatable
    # tf.truncated_normal() Outputs random values from truncated normal distro
    # (shape[], mean, stddev, dtype, seed, name)
with tf.name_scope('biases'):
    b1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
    ## bias 1 for the input layer 
    b2 = tf.Variable(tf.constant(0.1), [numClasses])

## defining matrix maltiplilfication 
with tf.name_scope('hidden_layer'):
    hiddenLayerOutput = tf.matmul(X, w1) + b1 # malti for input and hidden
    hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput) # activation funciton, ReLu here
with tf.name_scope('output_layer'):
    finalOutput = tf.matmul(hiddenLayerOutput, w2) + b2
    finalOutput = tf.nn.relu(finalOutput) # output labels

'''
the final 10 dimensional output will be fed through a softmax layer to obtain probabilities. Instead of defining the softmax function explicity, we'll be using Tensorflow's function tf.nn.softmax_cross_entropy_with_logits function to perform the softmax + the cross entropy loss all in one line.'''
#Loss (cost) function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))
# tf.nn.softmax...logits computes softmax cross entropy bet logits and labels
# label = label, logits = predicted y vale. 

with tf.name_scope('Train'):
    opt = tf.train.GradientDescentOptimizer(learning_rate = .005).minimize(loss)
# setting up operation, gradDescdent, and minimize takes cost function 

#### setting up parameter for prediction and acuraccy 
'''
The following are the variables that help with calculating accuracy. The correct prediction variable will take a look at the output vector, find the probability with the highest value, and return its index. This is then compared with the actual lables.
'''
correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y,1))
# tf.equal returns the truth value of (x==y), element-wise
# tf.argmax returns the index with the largest value across dimensions of a tensor

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# compute the average value of true/false or 1/0 across the examples, y label
# tf.cast, casts a tensor to a new type


########## TRAINING #######
'''

Here's the part where we start training our model. We'll load in our training set
by calling Tensorflow's mnist.train.next_batch() function. 
The sess.run function has two arguments. The first is called the 
"fetches" argument. It defines the value for you're interested in computing/running. 
For example, in our case, we want to both run the optimizer object so that 
the cross entropy loss gets minimized and evaluate the current loss value 
(in case we want to print that value). Therefore, we'll use [opt, loss] for our first argument. 
The second argument is where we input our feed_dict. This data structure is where we provide values to all of our placeholders. We repeat this for a set number of iterations.'''

trainAccuracy = 0
# create a summary for our cost and accuracy
cost_sum = tf.summary.scalar('cost', loss)
accuracy_sum = tf.summary.scalar('accuracy', trainAccuracy)
#summary_op = tf.summary.merge_all()

# initializing the session
sess = tf.Session()
init = tf.global_variables_initializer() # initializing all the vars
sess.run(init)
i = 0 # loop counter
## initializing the file writer
path = '/home/wataru/machineLearn/kaggle/TF_practice/testGraph2'
fwriter = tf.summary.FileWriter(path, graph=tf.get_default_graph())

while trainAccuracy <= 0.999:
    batch = mnist.train.next_batch(batchSize) # tuple containing X and y, numpy array
    batchInput = batch[0]
    batchLabels = batch[-1]
    _, trainingLoss, sumC = sess.run([opt, loss, cost_sum], feed_dict={X: batchInput, y: batchLabels})
    _, 

    # write log
    fwriter.add_summary(sumC, i*batchSize)   
    
    if i%1000 == 0:
        #trainAccuracy = accuracy.eval(session = sess, feed_dict = {X: batchInput, y: batchLabels})
        trainAccuracy, sumA = sess.run([accuracy, accuracy_sum], feed_dict={X:batchInput, y:batchLabels})
        fwriter.add_summary(sumA, i)
        print('step %d, training accuracy %g'%(i, trainAccuracy))
    i += 1


### TESTING ###
# testing the NN on test dataset
# loading the test dataset from mnist test images
testInputs = mnist.test.images ## both ndarrray of images, and labels
testLabels = mnist.test.labels
acc = accuracy.eval(session = sess, feed_dict={X: testInputs, y: testLabels})
print("testign accuracy: {}", format(acc))

# summary of the session and graph
