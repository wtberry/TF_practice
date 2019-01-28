import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

a = tf.constant(5.5)
b = tf.constant(8.6)

sess = tf.InteractiveSession()
print(sess.run(a))
print("sub a from b:", sess.run(tf.substract(b,a))) ## tf.add, multiply, or divide

## matrices

#matrices ops, like numpy
#tf.random_normal()
#tf.random_uniform()
#tf.ones()
#tf.zeros()
#tf.ones_like() generate a matrix of ones with the same shape as the matrix passed in

#### variable needs to be initialized upon using them by
# tf.global_variables_initializer()

### dot product / multiplication

# sess.run(tf.matmul(matrix11, matrix2))

### placeholders 
''' 
can define a series of computations without knowing the actual data, it'll still run
'''
c = tf.placeholder(tf.float32) # specifyign the datatype

### sess.run has two args,
## one for operations, such as c + d, and 
# the second is feed_dictionary where we input the actual value of placeholders as dictionary
# eg  sess.run(c + d, feed_dict = {c:4.0, d:5.0})
