import numpy as np
import tensorflow as tf





data = np.loadtxt("data2.txt")


X_train = data[0:20,0:4]
Y_train = data[0:20,4]

X_test = data[20:27,0:4]
Y_test = data[20:27,4]





tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.00005

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None,])

W1 = tf.Variable(tf.random_normal([4, 50],stddev=0.01), name='weight1')
b1 = tf.Variable(tf.random_normal([50], stddev=0.01), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([50, 30],stddev=0.01), name='weight2')
b2 = tf.Variable(tf.random_normal([30], stddev=0.01), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([30, 1],stddev = 0.01), name='weight3')
b3 = tf.Variable(tf.random_normal([1], stddev=0.01), name='bias3')
hypothesis = tf.nn.tanh(tf.matmul(layer2, W3) + b3)

# cost/loss function
cost = tf.reduce_mean((Y-hypothesis)*(Y-hypothesis))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
            
    for step in range(10000):
        idx = step % 20;
        sess.run(train, feed_dict={X: X_train[idx: idx + 1,:], Y: Y_train[idx:idx+1]})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: X_train, Y: Y_train})) 
                                                        
                                                            
                                                                                        

    result = sess.run(hypothesis, feed_dict={X : X_test})
    print(result)
    print(Y_test)


