import tensorflow as tf
import numpy as np
import os
import sys
import my_lib 
import time 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

 
start = time.time()

 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('_test.py')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)


sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))

u = sess.graph.get_tensor_by_name("u:0")
t = sess.graph.get_tensor_by_name("t:0")
isTrain = sess.graph.get_tensor_by_name("isTrain:0")

G_y = sess.graph.get_tensor_by_name("G_y:0")

mse = sess.graph.get_tensor_by_name("mse:0")
content_loss = sess.graph.get_tensor_by_name("content_loss:0")

content_optim = sess.graph.get_operation_by_name('content_optim')



np.random.seed(int(time.time()))

test_images = mnist.test.images[0:16]    
test_origin = test_images*0.9

temp_noise = np.random.uniform(size = (16,784))

test_input = np.minimum(test_images*0.4  + temp_noise, 1.0)

my_lib.mnist_4by4_save(np.reshape(test_origin,(-1,784)),file_name + '/ground_true.png') 
 


for i in range(4,9) :
    
 
    temp_noise = np.random.uniform(size = (16,784))

    test_input = np.minimum(test_images*i/10.0  + temp_noise, 1.0)

    my_lib.mnist_4by4_save(np.reshape(test_input,(-1,784)),file_name + '/result_noise_0_{}.png'.format(str(i).zfill(3)))
        
        
    r = sess.run([G_y,],feed_dict={u : np.reshape(test_input,(-1,28,28,1)),
        t : np.reshape(test_origin,(-1,28,28,1)),isTrain : False})
  
     
    my_lib.mnist_4by4_save(np.reshape(test_images*i/10.0,(-1,784)),file_name + '/result_input_0_{}.png'.format(str(i).zfill(3)))
    my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_0_{}.png'.format(str(i).zfill(3)))
 







