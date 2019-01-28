def train(batch_size, epochs):
    '''
    this takes batch size and epochs as args, and train CNN on fminst dataset
    '''
    ## tutorial by adesphande3 on github
    
    import tensorflow as tf
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    import fashion_data_import as fin
    #%matplotlib inline # what's this do?
    
    # loading MNIST dataset from TF library
    from tensorflow.examples.tutorials.mnist import input_data
    from sklearn import preprocessing
    #mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) # alraedy onehotted
    # but this should be the fMNIST dataset loading and data preprocessing 
    Xo, yo = fin.data_in('train')
    Xt, yt = fin.data_in('test')
    # oneHot
    enc = preprocessing.OneHotEncoder()
    enc.fit(yo)
    y_data = enc.transform(yo).toarray() # oneHot as ndarray
    #y_data = tf.one_hot(yo, depth=10) # oneHot as Tensor
    #yt_data = tf.one_hot(yt, depth=10)
    #enc = preprocessing.OneHotEncoder()
    enc.fit(yt)
    yt_data = enc.transform(yt).toarray() # oneHot as ndarray
    ## batch creating function
    def next_batch(num, data, labels):
        """
        Returns a total of 'num' random samples and labels
        """
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
    
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    ## Human readable labels, corresponding to the number labels
    human_label = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    
    ##### INPUTS AND OUTPUTS
    
    """
    So, in this next step, we're just going to create a session. 
    Your x and y_ are just going to place placeholders that basically just indicate
    the type of input you want in your CNN and the type of output. 
    For each of these placeholders, you have to specify the type and the shape.
    """
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape = [None, 28, 28, 1]) # shape in CNNs is always None x height x w
    y_ = tf.placeholder("float", shape=[None, 10]) # shape is always None x number of classes
    
    ##### NETWORK ARCHITECTURE #####
    """
    With placeholders, we can now specify the network architecture.
    All of the weights and filters are TF variable. 
    """
    # Filters and biases for the first layer
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)) # shape is fileter x filter x input channels x output channels
    b_conv1 = tf.Variable(tf.constant(.1, shape=[32])) # shape of the bias just has to match output channels of the filter
    
    ## Call first conv layer, with 4 vars. 
    # input(placeholder), the filter, the stride, the padding
    
    print('x', x)
    print('1st layer weight: ', W_conv1)
    
    h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    #h_conv1 = tf.nn.relu(h_conv1)
    h_conv1 = tf.nn.sigmoid(h_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ## organizing method calling functions
    def conv2d(x, W):
        return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ## completing the network
    
    # Second Conv and Pool Layers
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(.1, shape=[64]))
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    #First Fully Connected Layer
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(.1, shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout Layer
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Second Fully Connected Layer
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(.1, shape=[10]))
    
    # Final Layer
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    ## formulating loss function
    crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y))
    
    # optimizer to minimize the function
    trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
    
    ## Calculating the accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #y_, ?? 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    ## initializing everything
    sess.run(tf.global_variables_initializer())
    
    ## Visualization
    tf.summary.scalar('Cross_Entropy_Loss', crossEntropyLoss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    
    b, v = next_batch(1, Xo, y_data)
    print(b[0].shape) #b[0] contains the image
    image = tf.reshape(b[0], [-1, 28, 28, 1])
    print(image)
    my_img = image.eval() # here is your image Tensor
    my_i = my_img.squeeze()
    v_label = int(v.argmax(axis=1))
    xlabel = 'Label: ' + str(v_label) + ', ' + str(human_label[v_label])
    plt.xlabel(xlabel)
    plt.imshow(my_i, cmap='gray_r')
    plt.show()
    
    ##### TRAINING #####
    
    i = 0
    batchSize = batch_size 
    trainAccuracy = 0
    for i in range(epochs):
    #while trainAccuracy < 0.95 and i < 6000:
        #batch = mnist.train.next_batch(batchSize)
        X_batch, y_batch = next_batch(batchSize, Xo, y_data)
        trainingInputs = X_batch.reshape([batchSize, 28, 28, 1]) # 50x28x28x1
        trainingLabels = y_batch 
        #trainingInputs = batch[0].reshape([batchSize, 28, 28, 1]) # 50x28x28x1
        #trainingLabels = batch[1] # 50x10
        if i%10 == 0:
            summary = sess.run(merged, {x: trainingInputs, y_: trainingLabels, keep_prob:1.0})
            writer.add_summary(summary, i)
        if i%100 == 0:
            trainAccuracy = accuracy.eval(session=sess, feed_dict={x:trainingInputs, y_:trainingLabels, keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, trainAccuracy))
        trainStep.run(session = sess, feed_dict={x: trainingInputs, y_:trainingLabels,keep_prob:0.5})
        #i += 1 # increment the counter
    
    
    ##### TESTING #####
    
    #testInputs = mnist.test.images.reshape([-1, 28, 28, 1])
    #testLabels = mnist.test.labels
    testInputs = Xt.reshape([-1, 28, 28, 1]) # =1 becasue of placeholder setup for batch
    testLabels = yt_data
    acc = accuracy.eval(feed_dict = {x:testInputs, y_:testLabels, keep_prob:1.0})
    print("testing accuracy: {}".format(acc))

