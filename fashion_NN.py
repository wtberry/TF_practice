import tensorflow as tf
import numpy as np
import fashion_data_import as fin
from sklearn import preprocessing
import time
enc = preprocessing.OneHotEncoder()

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

input_layer_size = 784
hidden_layer_size =1000
num_labels = 10
trainingIterations = 60000
## converting the ndarray into Tensor
#X_data = tf.convert_to_tensor(Xo)
#y_data = tf.convert_to_tensor(yo)
#
## placeholder for in/outputs
tf.reset_default_graph()

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, input_layer_size], name='X-input')
    y = tf.placeholder(tf.float32, shape=[None, num_labels], name='y_input')



## defining weights and biases
with tf.name_scope('weights'):
    w1 = tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels], stddev=0.1))

with tf.name_scope('biases'):
    b1 = tf.Variable(tf.constant(0.1), [hidden_layer_size])
    b2 = tf.Variable(tf.constant(0.1), [num_labels]) 

## defining matrix maltiplification
# hidden layer calculation
with tf.name_scope('hidden_layer'):
    hiddenLayerOutput = tf.matmul(X, w1) + b1 # (none x ILS) x (ILS x HLS) = (nonex HLS) + HLS
    #hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
    hiddenLayerOutput = tf.nn.sigmoid(hiddenLayerOutput)

# output layer calculation
with tf.name_scope('output_layer'):
    finalOutput = tf.matmul(hiddenLayerOutput, w2) + b2
    #finalOutput = tf.nn.relu(finalOutput)
    finalOutput = tf.nn.sigmoid(finalOutput)

### Cost function!!!

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))

# softmax == backpropagation??
# reduce mean computes the average? of the all the errors

with tf.name_scope('Train'):
    ## optimization algorithm, grad descent in this case
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)
    #opt = tf.train.RMSPropOptimizer(learning_rate=0.9, decay=0.9).minimize(loss)
    #opt = tf.train.AdadeltaOptimizer(learning_rate=0.09).minimize(loss)

### setting up param for prediction and accuracy

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))



# Defining input_fn for training and testing
train_in = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(Xo)},
        y=np.array(y_data), num_epochs=None, shuffle=True, batch_size=500)
test_in = tf.estimator.inputs.numpy_input_fn(x={'x':np.array(Xt)},
        y=np.array(ytest), num_epochs=1, shuffle=False)
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


# initializing the session

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    i = 1 # loop counter

    ## initializing the file writer
    path = '/home/wataru/machineLearn/kaggle/TF_practice/tgf1'
    fwriter = tf.summary.FileWriter(path, graph=tf.get_default_graph())
    s_time = time.time() # initializing stopwatch
    while trainAccuracy <= 0.9 and i<=trainingIterations:
        ## defining batch to feed
        #X_batch, y_batch = next_batch(1000, Xo, y_data)        
        X_batch, y_batch = train_in()['x']
        #_, trainLoss, sumc = sess.run([opt, loss, s_cost], feed_dict={X:Xo, y:y_data})
        _, trainLoss, sumc = sess.run([opt, loss, s_cost], feed_dict={X:X_batch, y:y_batch})

        #write log
        fwriter.add_summary(sumc, i)

        if i%100 == 0:
            trainAccuracy, suma = sess.run([accuracy, s_accuracy], feed_dict={X:train_in.eval(), y:t})
            #trainAccuracy, suma = sess.run([accuracy, s_accuracy], feed_dict={X:X_batch, y:y_batch})
            fwriter.add_summary(suma, i)
            print('step %d, training accuracy %g'%(i, trainAccuracy))

        i += 1


###Testesting 

    testInputs = Xt
    testLabels = ytest
    acc = accuracy.eval(session = sess, feed_dict={X: testInputs, y:testLabels})
    print('testing accuracy: {}', format(acc))
    e_time = time.time() - s_time
    minutes = e_time//60
    seconds = e_time%60
    print(f"or.. as \n time elapsed: {minutes!r} min, {seconds!r} s")




