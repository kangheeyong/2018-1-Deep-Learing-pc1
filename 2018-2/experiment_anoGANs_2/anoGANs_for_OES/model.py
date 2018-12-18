import tensorflow as tf
import numpy as np
import os
import time
from anoGANs_for_OES.network import G, D
from anoGANs_for_OES.utility import idx_shuffle, mnist_4by4_save, gan_loss_graph_save, my_roc_curve, my_hist, mnist_matlab_4by4_save
from sklearn.metrics.pairwise import cosine_similarity

def MSE(x1,x2,name) :
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((x1-x2)**2, axis=[1,2,3])) , name = name) 
def Cross_Entropy(t,y,name) : 
    return tf.reduce_mean(tf.reduce_sum(-t*tf.log(y + 1e-8), axis = [1,2,3]),name = name)

class anoGANs_for_OES : 
    
    def __init__(self,sess, path, data, GANs_epoch = 100,E_epoch = 30, batch_size = 100, z_size = 100, 
                 G_lr = 2e-4, G_beta1 = 0.5, E_lr = 2e-4, E_beta1 = 0.1, D_lr = 2e-5, D_beta1 = 0.5, feature_size = 100,
                 minibatch_increase = False, factor_decrease = False) :
        
        self.sess = sess
        self.GANs_epoch = GANs_epoch
        self.E_epoch = E_epoch
        self.batch_size = batch_size
        self.z_size = z_size
        self.G_lr = G_lr
        self.G_beta1 = G_beta1
        self.E_lr = E_lr
        self.E_beta1 = E_beta1
        self.D_lr = D_lr
        self.D_beta1 = D_beta1
        self.path = path
        self.data = data
        self.feature_size = feature_size
        self.minibatch_increase = minibatch_increase
        self.factor_decrease = factor_decrease
        
        if not os.path.isdir(path) :
            os.mkdir(path)
            
        if not os.path.isdir(self.path+'/GANs_result') :
            os.mkdir(self.path+'/GANs_result')
        if not os.path.isdir(self.path+'/E_result') :
            os.mkdir(self.path+'/E_result')
        
        

        self.z = tf.placeholder(tf.float32,shape=(None,1,1,self.z_size),name = 'z')    
        self.u = tf.placeholder(tf.float32, shape = (None, 128,128,1),name='u')     
        self.isTrain = tf.placeholder(dtype=tf.bool,name='isTrain')
        self.factor = tf.placeholder(tf.float32, name = 'factor')

        self.G_z = G(self.z, self.isTrain, name='G_z') 

     
               
        self.D_real = D(self.u, self.isTrain, name = 'D_real')
        self.D_fake = D(self.G_z ,self.isTrain,reuse=True, name = 'D_fake')
      

        self.D_real_loss = tf.reduce_mean(-tf.log(self.D_real + 1e-8),name = 'D_real_loss')
        self.D_fake_loss = tf.reduce_mean(-tf.log(1 - self.D_fake + 1e-8),name = 'D_fake_loss')
                    

        self.D_loss =  tf.add(self.D_real_loss,self.D_fake_loss,name='D_loss')
        self.G_loss =  tf.reduce_mean(-tf.log(self.D_fake + 1e-8),name='G_loss')

        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('D')]
        G_vars = [var for var in T_vars if var.name.startswith('G')]



        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :  
            self.D_optim = tf.train.AdamOptimizer(self.factor*D_lr,beta1=D_beta1).minimize(self.D_loss, var_list=D_vars, name='D_optim') 
            self.G_optim = tf.train.AdamOptimizer(self.factor*G_lr,beta1=G_beta1).minimize(self.G_loss, var_list=G_vars, name='G_optim')

            
    def GANs_fit(self) :
        tf.set_random_seed(int(time.time()))
        self.sess.run(tf.global_variables_initializer())


        test_z = np.random.uniform(-1,1,size=(16,1,1,self.z_size))
        mnist_4by4_save(np.reshape(self.data.test_normal_data[0:16],(-1,128,128,1)),self.path + '/GANs_result/normal')    
        mnist_matlab_4by4_save(np.reshape(self.data.test_normal_data[0:16],(-1,128,128,1)),self.path + '/GANs_result/normal')    
        mnist_4by4_save(np.reshape(self.data.test_anomalous_data[0:16],(-1,128,128,1)),self.path + '/GANs_result/anomalous')    
        mnist_matlab_4by4_save(np.reshape(self.data.test_anomalous_data[0:16],(-1,128,128,1)),self.path + '/GANs_result/anomalous') 
        log_txt = open(self.path +'/GANs_result/log.txt','w')

        hist_G = []
        hist_D = []
        hist_measure = []
        G_error = []
        D_error = []
        D_fake_error = []
        D_real_error = []


        start = time.time()
        for epoch in range(self.GANs_epoch) :
            self.data.train_normal_data = idx_shuffle(self.data.train_normal_data) 
            
            if ((epoch < 1) & self.minibatch_increase)  : 
                temp_batch = int(self.batch_size/4)

            elif ((epoch < 2) & self.minibatch_increase)  : 
                temp_batch = int(self.batch_size/2)

            else : 
                temp_batch = self.batch_size

            
            if ((epoch > self.GANs_epoch*0.66) & self.factor_decrease)  :
                factor = 0.01           
            elif ((epoch > self.GANs_epoch*0.33) & self.factor_decrease)  :
                factor = 0.1
            else :
                factor = 1

            for iteration in range(self.data.train_normal_data.shape[0] // temp_batch) : 
                
                train_images = self.data.train_normal_data[iteration*temp_batch : (iteration+1)*temp_batch]      
                u_ = np.reshape(train_images,(-1,128,128,1)) 
                z_ = np.random.uniform(-1,1,size=(temp_batch,1,1,self.z_size))                                                                                               

                _ , D_e, D_real_e, D_fake_e = self.sess.run([self.D_optim, self.D_loss, self.D_real_loss, self.D_fake_loss],
                                                     {self.u : u_, self.z : z_, self.isTrain : True, self.factor : factor})

                _ , G_e = self.sess.run([self.G_optim, self.G_loss],
                                       {self.u : u_, self.z : z_, self.isTrain : True, self.factor : factor}) 


                D_error.append(D_e)
                D_real_error.append(np.maximum(0.0, D_real_e))
                D_fake_error.append(np.maximum(0.0,D_fake_e))
                G_error.append(G_e)


            hist_D.append(np.mean(D_error)) 
            hist_G.append(np.mean(G_error))


            print('D_e : %.3f, D_real_e : %.3f, D_fake_e : %.3f, G_e : %.3f\n'
                  %(np.mean(D_error), np.mean(D_real_error),np.mean(D_fake_error), np.mean(G_error)))
            log_txt.write('D_e : %.3f, D_real_e : %.3f, D_fake_e : %.3f, G_e : %.3f\n'
                          %(np.mean(D_error), np.mean(D_real_error),np.mean(D_fake_error), np.mean(G_error)))

            r = self.sess.run([self.G_z],feed_dict={self.z : test_z, self.isTrain : False})       
            mnist_4by4_save(np.reshape(r,(-1,128,128,1)),self.path + '/GANs_result/G_{}'.format(str(epoch).zfill(3)))
            mnist_matlab_4by4_save(np.reshape(r,(-1,128,128,1)),self.path + '/GANs_result/G_{}'.format(str(epoch).zfill(3)))
            
            np.random.seed(int(time.time()))


            G_error = []
            D_error = []       
            D_fake_error = []     
            D_real_error = []




        gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = self.path + '/GANs_result/loss_graph.png')   
        saver = tf.train.Saver()
        saver.save(self.sess,self.path + '/GANs_result/para.cktp')

        end = time.time()-start

        print("total time : ",end)
        log_txt.write("total time : %d\n"%(end))
        log_txt.close()
        
        

        
        
        







