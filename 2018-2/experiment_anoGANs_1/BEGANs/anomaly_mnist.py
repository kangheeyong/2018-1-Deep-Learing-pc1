
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data



class Anomaly_Mnist :
    def __init__(self) :
        
        self.train_normal_data = []
        self.train_anomalous_data = []
        self.test_normal_data = []
        self.test_anomalous_data = []
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,reshape=[])
        
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess :
            sess.run(tf.global_variables_initializer())

            self.train_set = tf.image.resize_images(mnist.train.images,[64,64]).eval()                                                                                                           
            self.train_label = mnist.train.labels
            self.train_set = (self.train_set -0.5)/0.5

            self.test_set = tf.image.resize_images(mnist.test.images,[64,64]).eval()                                                                                                           
            self.test_label = mnist.test.labels
            self.test_set = (self.test_set -0.5)/0.5
            
  
    def set_anomaly(self, anomalous_nums) :
            train_size = 55000
            test_size = 10000
            
            s1 = set(anomalous_nums)
            
            
            for i in range(train_size) :
                
                s2 = set([np.argmax(self.train_label[i])])
                if   s2 != (s1 & s2) :
                    self.train_normal_data.append(self.train_set[i])
                else : 
                    self.train_anomalous_data.append(self.train_set[i])

            self.train_normal_data = np.array(self.train_normal_data)        
            self.train_anomalous_data = np.array(self.train_anomalous_data)        

            for i in range(test_size) :

                s2 = set([np.argmax(self.test_label[i])])
                if   s2 != (s1 & s2) :
                    self.test_normal_data.append(self.test_set[i])
                else : 
                    self.test_anomalous_data.append(self.test_set[i])        

            self.test_normal_data = np.array(self.test_normal_data)        
            self.test_anomalous_data = np.array(self.test_anomalous_data)              

            print("anomalous number : ", anomalous_nums)
            print('test_normal_data : ' ,self.test_normal_data.shape)
            print('test_anomalous_data : ' ,self.test_anomalous_data.shape)
            print('train_normal_data : ' ,self.train_normal_data.shape)
            print('train_anomalous_data : ' ,self.train_anomalous_data.shape)
        

