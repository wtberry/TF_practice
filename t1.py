# first tf tutorial on constants and session
import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)

print(node1, node2)

### process are run inside of session
sess = tf.Session() # lanch the graph and run 
#print(sess.run([node1, node2]))

#sess.close() # closing the session

with tf.Session() as sess: ## session as code block 
    output = sess.run([node1, node2])
    print(output)

with tf.Session() as sess2:
    a = tf.constant(3.0)
    b = tf.constant(5.3)
    c = a*b
    result = sess2.run(c)
    print(result)

##### graph visualization #####

# use tensorboard

path_to_theoutput = '/home/wataru/machineLearn/kaggle/TF_practice/testGraph'
file_writer = tf.summary.FileWriter(path_to_theoutput, sess2.graph)
