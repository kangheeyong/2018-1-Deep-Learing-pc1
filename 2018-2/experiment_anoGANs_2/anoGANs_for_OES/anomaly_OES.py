
import numpy as np
import tensorflow as tf
import pickle
import gzip
from tensorflow.examples.tutorials.mnist import input_data



class Anomaly_OES :
    def __init__(self) :

        
        dir_file = 'normal/'
        with gzip.open(dir_file  + 'normal_data.pickle.gzip','rb') as f :
            self.normal_data = (np.array(pickle.load(f))-2000)/7000
        
        
        dir_file = 'anomaly/'
        with gzip.open(dir_file  + 'anomaly_data.pickle.gzip','rb') as f :
            self.anomaly_data = (np.array(pickle.load(f))-2000)/7000
            
  
    def set_anomaly(self) :
        self.train_normal_data = self.normal_data[0 : self.normal_data.shape[0] - self.anomaly_data.shape[0]]
        #self.train_anomalous_data = []
        self.test_normal_data = self.normal_data[self.normal_data.shape[0] - self.anomaly_data.shape[0] : self.normal_data.shape[0]]
        self.test_anomalous_data = self.anomaly_data

            
               


        print('test_normal_data : ' ,self.test_normal_data.shape)
        print('test_anomalous_data : ' ,self.test_anomalous_data.shape)
        print('train_normal_data : ' ,self.train_normal_data.shape)
        #print('train_anomalous_data : ' ,self.train_anomalous_data.shape)
        

