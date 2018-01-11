import tensorflow as tf
import numpy as np
import os
import sys
import time
import my_lib 
import time 

os.environ["CUDA_VISIBLE_DEVICES"]="0"



start = time.time()

  
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


file_name = sys.argv[0].split('_load.py')[0]

if not os.path.isdir(file_name) :
    os.mkdir(file_name)
    
sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(file_name + '/para.cktp.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(file_name + '/'))
    
 
one_hot = np.eye(10)



u = sess.graph.get_tensor_by_name("u:0")
z_c = sess.graph.get_tensor_by_name("z_c:0")
z_fill = sess.graph.get_tensor_by_name("z_fill:0")
isTrain = sess.graph.get_tensor_by_name("isTrain:0")

    
G_z = sess.graph.get_tensor_by_name("G_z:0")

D_real_loss = sess.graph.get_tensor_by_name('D_real_loss:0')
D_fake_loss = sess.graph.get_tensor_by_name('D_fake_loss:0')

D_loss = sess.graph.get_tensor_by_name("D_loss:0")
G_loss = sess.graph.get_tensor_by_name("G_loss:0")

D_optim = sess.graph.get_operation_by_name("D_optim")
G_optim = sess.graph.get_operation_by_name("G_optim")


np.random.seed(int(time.time()))

test_z = np.random.normal(0,1,size=(16,1,1,100))
test_temp = np.random.randint(0,9,(16,1))
test_z_c = one_hot[test_temp].reshape([-1,1,1,10])
np.savetxt(file_name+'/condition.txt',test_temp,fmt='%d')

log_txt = open(file_name +'/log.txt','w')

hist_G = []
hist_D = []
G_error = []
D_error = []
D_fake_error = []
D_real_error = []
for i in range(100000) :


    train_images,train_labels = mnist.train.next_batch(100)    
    z_c_ = np.reshape(train_labels,(-1,1,1,10))    
    z_fill_ = z_c_*np.ones([100,28,28,10])
    u_ = np.reshape(train_images,(-1,28,28,1)) 
    z_ = np.random.normal(0,1,size=(100,1,1,100))


    _ , D_e,D_real_e,D_fake_e = sess.run([D_optim, D_loss,D_real_loss,D_fake_loss], {u : u_,z_fill : z_fill_, z : z_,z_c : z_c_, isTrain : True})
    D_error.append(D_e)
    D_real_error.append(D_real_e)
    D_fake_error.append(D_fake_e)
#    train_images,train_labels = mnist.train.next_batch(100)    
#    z_c_ = np.reshape(train_labels,(-1,1,1,10))    
#    z_fill_ = z_c_*np.ones([100,28,28,10])
#    u_ = np.reshape(train_images,(-1,28,28,1)) 
#    z_ = np.random.normal(0,1,size=(100,1,1,100))

    _ , G_e = sess.run([G_optim, G_loss], {u : u_,z_fill : z_fill_, z : z_,z_c : z_c_, isTrain : True}) 
    G_error.append(G_e)
        

    if i%1000 == 0:

        hist_D.append(np.mean(D_error)) 
        hist_G.append(np.mean(G_error))

        print('D_e : %.6f, D_real_e : %.6f, D_fake_e : %.6f, G_e : %.6f'%(np.mean(D_error), np.mean(D_real_error),
            np.mean(D_fake_error), np.mean(G_error)))
        log_txt.write('D_e : %.6f, D_real_e : %.6f, D_fake_e : %.6f, G_e : %.6f\n'%(np.mean(D_error),
            np.mean(D_real_error), np.mean(D_fake_error), np.mean(G_error)))
      
        r = sess.run([G_z],feed_dict={z : test_z,z_c : test_z_c, isTrain : False})        
        my_lib.mnist_4by4_save(np.reshape(r,(-1,784)),file_name + '/result_{}.png'.format(str(i).zfill(3)))

        np.random.seed(int(time.time()))

        G_errer = []
        D_errer = []
        D_fake_error = []
        D_real_error = []


log_txt.close()
my_lib.gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = file_name + '/loss_graph.png')   
saver = tf.train.Saver()
saver.save(sess,file_name + '/para.cktp')


end = time.time()-start

print("total time : ",end)






