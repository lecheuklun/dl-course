import numpy as np
import tensorflow as tf

coefficient = np.array([[1.], [-10.], [25.]])

# set up
w = tf.Variable(0,dtype=tf.float32) # parameter to optimise
x = tf.placeholder(tf.float32, [3,1])

# cost = tf.add(w**2,tf.multiplpy(-10.,w)),25)
cost = x[0][0]*w**2 - x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # learning rate

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w)) # 0.0

session.run(train, feed_dict={x:coefficients}) # one step of GD
print(session,run(w))
0.1

for i in range(1000):
	session.run(train, feed_dict={x:coefficients})
print(session.run(w)) # 4.9999

