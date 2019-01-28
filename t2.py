## Plaaceholders, Variables and simple linear regression

import tensorflow as tf 

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
## place holder accepts external input, promise to provide values later
add_node = a+b

with tf.Session() as sess:
    print(sess.run(add_node, {a:[2.3, 4.3],b:[5.4, 12.0]})) 
    # putting values into placeholder, as dictionary and list

    path = '/home/wataru/machineLearn/kaggle/TF_practice/testGraph'
    fwriter = tf.summary.FileWriter(path, sess.graph) # write a summary of the sesson as graph

#### valuable, updatable nodes ######
## valuable can be updated each iteration, like weight

## initial value of line
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

## inputs and outputs
x = tf.placeholder(tf.float32) # input

linear_model = W*x + b
y = tf.placeholder(tf.float32) # labels

# cost function
squared_delta = tf.square(linear_model -y)
loss = tf.reduce_sum(squared_delta)

## optimize
optimizer = tf.train.GradientDescentOptimizer(0.01) # returns optimizer operation object, learning rate as arg
train = optimizer.minimize(loss) # takes cost function as arg


## initialize the all vars
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) #initialize all the variables
    #start running the algorithm
    # loop for gradient descent ephoc 
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0, -1, -2, -3]}) # running trainng 
        print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))
    #print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))
    print('W, B: ', sess.run([W, b]))
