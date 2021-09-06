import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

y_hat = tf.constant(36, name='y_hat') # prediction
y = tf.constant(39, name='y') 

loss = tf.Variable((y-y_hat)**2, name='loss')

init = tf.global_variables_initializer() # loss variable initialised, ready for computing

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

# using TF:
# 1 Create un-initialised Tensors (variables), 2 write operations between Tensors, 3 Initialise Tensors
# 4 Create sessions, 5 run session
# create vars, initialise, create session, run vars inside sess

x = tf.placeholder(tf.int64, name= 'x') # var x defined
print(sess.run(2*x, feed_dict = {x:3})) # feed 3->x
sess.close()

# linear functions, Y = WX+b

# sigmoid function

def sigmoid(z):
    x = tf.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
    return result

# cost function

def cost(logits, labels):
    # logits -> z, lables -> y
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()

    cost = session.run(cost, feed_dict={z:logits, y:labels})
    
    sess.close()

    return cost

