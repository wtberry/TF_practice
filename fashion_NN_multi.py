## 2 hidden layer version of fashion MNIST
import tensorflow as tf
import numpy as np
import fashion_data_import as fin
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
import time

## importing fashion MNIST dataset as ndarray

Xo, yo = fin.data_in('train')
m, n = Xo.shape
#convert y label vector to one-hot y matrix
enc.fit(yo)
y_data = enc.transform(yo).toarray() # ndarray

print('X\n', Xo)
print('y\n', y_data)

Xt, yt = fin.data_in('test')
mt, nt = Xt.shape
enc.fit(yt)
ytest = enc.transform(yt).toarray()

### parameter setup for NN
tf.reset_default_graph()

input_layer_size = 784
hidden_layer_size1=800
hidden_layer_size2 =800
num_labels = 10
learn_r = 0.1
## decaying learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.2
learn_r =tf.train.exponential_decay(starter_learning_rate, global_step, 8000, 0.90, staircase=True) 
batch_size = 1000
trainingIterations = 20000
print_num = 100
## converting the ndarray into Tensor
#X_data = tf.convert_to_tensor(Xo)
#y_data = tf.convert_to_tensor(yo)
#
## placeholder for in/outputs

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, input_layer_size], name='X-input')
    y = tf.placeholder(tf.float32, shape=[None, num_labels], name='y_input')



## defining weights and biases
with tf.name_scope('weights'):
    w1 = tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size1], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([hidden_layer_size1, hidden_layer_size2], stddev=0.1))
    w3 = tf.Variable(tf.truncated_normal([hidden_layer_size2, num_labels], stddev=.1))

with tf.name_scope('biases'):
    b1 = tf.Variable(tf.constant(0.1), [hidden_layer_size1])
    b2 = tf.Variable(tf.constant(0.1), [hidden_layer_size2]) 
    b3 = tf.Variable(tf.constant(0.1), [num_labels])

## defining matrix maltiplification
# hidden layer calculation
with tf.name_scope('hidden_layer1'):
    hiddenLayerOutput = tf.matmul(X, w1) + b1 # (none x ILS) x (ILS x HLS) = (nonex HLS) + HLS
    #hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
    hiddenLayerOutput = tf.nn.sigmoid(hiddenLayerOutput)
with tf.name_scope('hidden_layer2'):
    # hiddenLayer2
    hiddenLayer2Output = tf.matmul(hiddenLayerOutput, w2) + b2
    hiddenLayer2Output = tf.nn.sigmoid(hiddenLayer2Output)
    #hiddenLayer2Output = tf.nn.relu(hiddenLayer2Output)

# output layer calculation
with tf.name_scope('output_layer'):
    finalOutput = tf.matmul(hiddenLayer2Output, w3) + b3
    #finalOutput = tf.nn.relu(finalOutput)
    finalOutput = tf.nn.sigmoid(finalOutput)

### Cost function!!!

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))

# softmax == backpropagation
# reduce mean computes the average? of the all the errors

with tf.name_scope('Train'):
    ## optimization algorithm, grad descent in this case
    opt = tf.train.GradientDescentOptimizer(learning_rate=learn_r).minimize(loss)
    #opt = tf.train.AdadeltaOptimizer(learning_rate=0.09).minimize(loss)

### setting up param for prediction and accuracy

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))



## TRAINING!!!!

## defining batch creating function

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


trainAccuracy = 0 #setting it for 0 for now for graph and while loop
# summary for cost and accuracy
s_cost = tf.summary.scalar('cost', loss)
s_accuracy = tf.summary.scalar('accuracy', trainAccuracy)

## initializing stopwatch
start_time = time.time()
# initializing the session

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    i = 1 # loop counter

    ## initializing the file writer
    path = '/home/wataru/machineLearn/kaggle/TF_practice/tgf1'
    fwriter = tf.summary.FileWriter(path, graph=tf.get_default_graph())
    
    while trainAccuracy <= 0.95:
        ## defining batch to feed
        X_batch, y_batch = next_batch(batch_size, Xo, y_data)        
        #_, trainLoss, sumc = sess.run([opt, loss, s_cost], feed_dict={X:Xo, y:y_data})
        _, trainLoss, sumc = sess.run([opt, loss, s_cost], feed_dict={X:X_batch, y:y_batch})

        #write log
        fwriter.add_summary(sumc, i)

        if i%print_num == 0:
            trainAccuracy, suma = sess.run([accuracy, s_accuracy], feed_dict={X:Xo, y:y_data})
            #trainAccuracy, suma = sess.run([accuracy, s_accuracy], feed_dict={X:X_batch, y:y_batch})
            fwriter.add_summary(suma, i)
            print('step %d, training accuracy %g'%(i, trainAccuracy))
            print('Learning rate: ', learn_r.eval())

        i += 1


###Testesting 

    testInputs = Xt
    testLabels = ytest
    acc = accuracy.eval(session = sess, feed_dict={X: testInputs, y:testLabels})
    e_time = time.time() - start_time
    minutes = e_time//60
    seconds = e_time%60
    print('testing accuracy: {}', format(acc))
    param = {'input_size': input_layer_size, 'hidden_layer_size1':hidden_layer_size1, 'hidden_layer_size2':hidden_layer_size2, 'learning_rate':learn_r, 'batch_size': batch_size}
    print(param)
    print('training time: ', minutes, 'min,', seconds, 'sec')




