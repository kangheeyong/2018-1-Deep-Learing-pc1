import sys



file_name = sys.argv[1]
ano_nums = [int (i) for i in sys.argv[2].split()]
gpu_num = sys.argv[3]


print(file_name)

print(ano_nums)

print(gpu_num)

import matplotlib
matplotlib.use('Agg')
from anoGANs import Anomaly_Mnist
from anoGANs import BE_infoGANs_v2
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

data = Anomaly_Mnist()
data.set_anomaly(anomalous_nums = ano_nums)



sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) 


path = file_name
mnist = data
a = BE_infoGANs_v2(sess,path,mnist, GANs_epoch = 100, E_epoch = 30)
a.GANs_fit()
a.E_fit()
a.report()








