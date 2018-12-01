import tensorflow as tf
import numpy as np
import os
import time
from anoGANs.network import G, E, D_enc, D_dec, Q_cat
from anoGANs.utility import idx_shuffle, mnist_4by4_save, gan_loss_graph_save, my_roc_curve, my_hist
from sklearn.metrics.pairwise import cosine_similarity

def MSE(x1,x2,name) :
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((x1-x2)**2, axis=[1,2,3])) , name = name) 
def Cross_Entropy(t,y,name) : 
    return tf.reduce_mean(tf.reduce_sum(-t*tf.log(y + 1e-8), axis = [1,2,3]),name = name)

class BE_infoGANs_v2 : 
    
    def __init__(self,sess, path, data, GANs_epoch = 100,E_epoch = 30, batch_size = 100, z_size = 100, lam = 0.01, gamma = 0.7, k_curr = 0.0,
                G_lr = 2e-4, G_beta1 = 0.5, E_lr = 2e-4, E_beta1 = 0.1, D_lr = 2e-5, D_beta1 = 0.5, c_size = 10, feature_size = 100, minibatch_increase = False) :
        
        self.sess = sess
        self.GANs_epoch = GANs_epoch
        self.E_epoch = E_epoch
        self.batch_size = batch_size
        self.z_size = z_size
        self.lam = lam
        self.gamma = gamma
        self.k_curr = k_curr
        self.G_lr = G_lr
        self.G_beta1 = G_beta1
        self.E_lr = E_lr
        self.E_beta1 = E_beta1
        self.D_lr = D_lr
        self.D_beta1 = D_beta1
        self.path = path
        self.data = data
        self.c_size = c_size
        self.feature_size = feature_size
        self.minibatch_increase = minibatch_increase
        
        if not os.path.isdir(path) :
            os.mkdir(path)
            
        if not os.path.isdir(self.path+'/GANs_result') :
            os.mkdir(self.path+'/GANs_result')
        if not os.path.isdir(self.path+'/E_result') :
            os.mkdir(self.path+'/E_result')
        
        

        self.z = tf.placeholder(tf.float32,shape=(None,1,1,self.z_size),name = 'z')    
        self.c = tf.placeholder(tf.float32,shape=(None,1,1,self.c_size),name = 'c')   
        self.u = tf.placeholder(tf.float32, shape = (None, 64,64,1),name='u')     
        self.k = tf.placeholder(tf.float32, name = 'k')
        self.isTrain = tf.placeholder(dtype=tf.bool,name='isTrain')  

        self.G_z = G(self.z, self.c, self.isTrain, name='G_z') 
        self.E_u,  self.E_u_c = E(self.u, self.isTrain,name = 'E_u', c_size = self.c_size, z_size = self.z_size) 

        self.re_image = G(self.E_u, self.E_u_c, self.isTrain, reuse=True, name ='re_image')
        self.re_z, self.re_z_c = E(self.G_z, self.isTrain, reuse=True, name ='re_z',c_size = self.c_size,z_size = self.z_size)

        self.re_z_loss = MSE(self.re_z , self.z, name = 're_z_loss') 
        self.re_z_c_loss = Cross_Entropy(self.c, self.re_z_c, name = 're_z_c_loss')
        self.re_image_loss = MSE(self.re_image,self.u, name = 're_image_loss') 

        self.E_loss = tf.add(self.re_z_loss, self.re_z_c_loss, name = 'E_loss')                       

        self.D_enc = D_enc(self.u, self.isTrain, name = 'D_enc')
        self.D_real = D_dec(self.D_enc, self.isTrain, name = 'D_real')                       
        self.D_fake = D_dec(D_enc(self.G_z, self.isTrain,reuse=True), self.isTrain, reuse=True, name = 'D_fake')         
        self.Q_fake = Q_cat(D_enc(self.G_z, self.isTrain,reuse=True), name='Q_fake', c_size = self.c_size)


        self.D_real_loss = MSE(self.D_real, self.u, name = 'D_real_loss')             
        self.D_fake_loss = MSE(self.D_fake, self.G_z,  name = 'D_fake_loss' )
        self.D_loss =  tf.add(self.D_real_loss, -self.k*self.D_fake_loss, name='D_loss')                                        

        self.G_loss =  MSE(self.D_fake , self.G_z, name='G_loss')                            
        self.Q_loss = Cross_Entropy(self.c, self.Q_fake, name = 'Q_loss')


        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('D_dec') or var.name.startswith('D_enc')]
        G_vars = [var for var in T_vars if var.name.startswith('G')]
        E_vars = [var for var in T_vars if var.name.startswith('E')]
        Q_vars = [var for var in T_vars if var.name.startswith('Q')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) :  
            self.D_optim = tf.train.AdamOptimizer(D_lr,beta1=D_beta1).minimize(self.D_loss, var_list=D_vars, name='D_optim') 
            self.G_optim = tf.train.AdamOptimizer(G_lr,beta1=G_beta1).minimize(self.G_loss + self.Q_loss, var_list=G_vars+Q_vars, name='G_optim')
            self.E_optim = tf.train.AdamOptimizer(E_lr,beta1=E_beta1).minimize(self.E_loss, var_list=E_vars, name='E_optim')
            self.E_AE_optim = tf.train.AdamOptimizer(E_lr,beta1=E_beta1).minimize(self.re_image_loss, var_list=E_vars, name='E_AE_optim')
            
    def GANs_fit(self) :
        tf.set_random_seed(int(time.time()))
        self.sess.run(tf.global_variables_initializer())

        one_hot = np.eye(self.c_size)
        test_c = one_hot[np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])].reshape([-1,1,1,self.c_size])
        test_z = np.random.uniform(-1,1,size=(16,1,1,self.z_size))
        mnist_4by4_save(np.reshape(self.data.test_normal_data[0:16],(-1,64,64,1)),self.path + '/GANs_result/normal.png')    
        mnist_4by4_save(np.reshape(self.data.test_anomalous_data[0:16],(-1,64,64,1)),self.path + '/GANs_result/anomalous.png')    

        log_txt = open(self.path +'/GANs_result/log.txt','w')

        hist_G = []
        hist_D = []
        hist_measure = []
        G_error = []
        D_error = []
        Q_error=[]
        D_fake_error = []
        D_real_error = []
        new_measure = []

        start = time.time()
        for epoch in range(self.GANs_epoch) :
            self.data.train_normal_data = idx_shuffle(self.data.train_normal_data) 
            
            if epoch < 3 and self.minibatch_increase  : 
                temp_batch = 32
            elif epoch < 5 and self.minibatch_increase  : 
                temp_batch = 64
            else : 
                temp_batch = self.batch_size
            
            
            for iteration in range(self.data.train_normal_data.shape[0] // temp_batch) : 

                train_images = self.data.train_normal_data[iteration*temp_batch : (iteration+1)*temp_batch]      
                u_ = np.reshape(train_images,(-1,64,64,1)) 
                z_ = np.random.uniform(-1,1,size=(temp_batch,1,1,self.z_size))                                                                                               
                c_ = one_hot[np.random.randint(0,self.c_size,(temp_batch))].reshape([-1,1,1,self.c_size])

                _ , D_e, D_real_e, D_fake_e = self.sess.run([self.D_optim, self.D_loss, self.D_real_loss, self.D_fake_loss],
                                                     {self.u : u_, self.z : z_, self.c : c_, self.k : self.k_curr, self.isTrain : True})

                _ , G_e,Q_e = self.sess.run([self.G_optim, self.G_loss,self.Q_loss],
                                       {self.u : u_, self.z : z_, self.c : c_, self.k : self.k_curr, self.isTrain : True}) 

                self.k_curr = self.k_curr + self.lam * (self.gamma*D_real_e - G_e)
                measure = D_real_e + np.abs(self.gamma*D_real_e - G_e)

                D_error.append(D_e)
                D_real_error.append(np.maximum(0.0, D_real_e))
                D_fake_error.append(np.maximum(0.0,D_fake_e))
                G_error.append(G_e)
                Q_error.append(Q_e)
                new_measure.append(measure)

            hist_D.append(np.mean(D_error)) 
            hist_G.append(np.mean(G_error))
            hist_measure.append(np.mean(new_measure))

            print('D_e : %.3f, D_real_e : %.3f, D_fake_e : %.3f, G_e : %.3f, Q_e : %.3f, new_measure : %.3f, k_curr : %.3f'
                  %(np.mean(D_error), np.mean(D_real_error),np.mean(D_fake_error), np.mean(G_error),
                    np.mean(Q_error),np.mean(new_measure),self.k_curr))
            log_txt.write('D_e : %.3f, D_real_e : %.3f, D_fake_e : %.3f, G_e : %.3f, Q_e : %.3f, new_measure : %.3f, k_curr : %.3f\n'
                  %(np.mean(D_error), np.mean(D_real_error),np.mean(D_fake_error), np.mean(G_error),
                    np.mean(Q_error),np.mean(new_measure),self.k_curr))

            r = self.sess.run([self.G_z],feed_dict={self.z : test_z, self.c : test_c, self.isTrain : False})       
            mnist_4by4_save(np.reshape(r,(-1,64,64,1)),self.path + '/GANs_result/G_{}.png'.format(str(epoch).zfill(3)))

            r = self.sess.run([self.D_real],feed_dict={self.u : self.data.test_normal_data[0:16], self.isTrain : False})        
            mnist_4by4_save(np.reshape(r,(-1,64,64,1)),self.path + '/GANs_result/D_{}.png'.format(str(epoch).zfill(3)))

            np.random.seed(int(time.time()))


            G_error = []
            D_error = []       
            D_fake_error = []     
            D_real_error = []
            new_measure = []



        gan_loss_graph_save(G_loss = hist_G,D_loss=hist_D,path = self.path + '/GANs_result/loss_graph.png')   
        saver = tf.train.Saver()
        saver.save(self.sess,self.path + '/GANs_result/para.cktp')

        end = time.time()-start

        print("total time : ",end)
        log_txt.write("total time : %d\n"%(end))
        log_txt.close()
        
        
    def E_fit(self) : 

        one_hot = np.eye(self.c_size)

        test_c = one_hot[np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])].reshape([-1,1,1,self.c_size])
        test_z = np.random.uniform(-1,1,size=(16,1,1,self.z_size))
        mnist_4by4_save(np.reshape(self.data.test_normal_data[0:16],(-1,64,64,1)),self.path + '/E_result/normal.png')    
        mnist_4by4_save(np.reshape(self.data.test_anomalous_data[0:16],(-1,64,64,1)),self.path + '/E_result/anomalous.png')    
        

        E_error = []
        log_txt = open(self.path +'/E_result/log.txt','w')
        start = time.time()
        for epoch in range(self.E_epoch) :

            np.random.seed(int(time.time()))
            self.data.train_normal_data = idx_shuffle(self.data.train_normal_data) 

            for iteration in range(self.data.train_normal_data.shape[0] // self.batch_size) : 


                train_images = self.data.train_normal_data[iteration*self.batch_size : (iteration+1)*self.batch_size]      
                u_ = np.reshape(train_images,(-1,64,64,1)) 
                z_ = np.random.uniform(-1,1,size=(self.batch_size,1,1,self.z_size))                                                           
                c_ = one_hot[np.random.randint(0,self.c_size,(self.batch_size))].reshape([-1,1,1,self.c_size])

                _  = self.sess.run([self.E_optim], {self.u : u_, self.z : z_, self.c : c_, self.isTrain : True})

                _ , E_e = self.sess.run([self.E_AE_optim, self.re_image_loss], {self.u : u_, self.z : z_, self.c : c_, self.isTrain : True})
                E_error.append(E_e)




            r = self.sess.run([self.re_image],feed_dict={self.u : self.data.test_normal_data[0:16],self.isTrain : False})        
            mnist_4by4_save(np.reshape(r,(-1,64,64,1)),self.path + '/E_result/normal_{}.png'.format(str(epoch).zfill(3)))

            r = self.sess.run([self.re_image],feed_dict={self.u : self.data.test_anomalous_data[0:16],self.isTrain : False})        
            mnist_4by4_save(np.reshape(r,(-1,64,64,1)),self.path + '/E_result//anomlous_{}.png'.format(str(epoch).zfill(3)))
            
            
            print("E_e : %.6f"%(np.mean(E_error)))
            log_txt.write("E_e : %.6f"%(np.mean(E_error)))
            E_error = []

        saver = tf.train.Saver()
        saver.save(self.sess,self.path + '/E_result/para.cktp')

        end = time.time()-start

        print("total time : ",end)
        log_txt.write("total time : %d\n"%(end))
        log_txt.close()
        
    def report(self) : 
        
        test_normal_feature_mse = []
        test_normal_feature_cosine = []
        test_normal_feature_signcode = []
        test_normal_residual_mse = []
        test_normal_residual_cosine = []
        
        test_anomalous_feature_mse = []
        test_anomalous_feature_cosine = []
        test_anomalous_feature_signcode = []
        test_anomalous_residual_mse = []
        test_anomalous_residual_cosine = []
        
        train_normal_feature_mse = []
        train_normal_feature_cosine = []
        train_normal_feature_signcode = []
        train_normal_residual_mse = []
        train_normal_residual_cosine = []

        for iteration in range(self.data.test_normal_data.shape[0] // self.batch_size) : 
            test_images = self.data.test_normal_data[iteration*self.batch_size : (iteration+1)*self.batch_size]      
            u_ = np.reshape(test_images,(-1,64,64,1)) 

            im_enc= self.sess.run([self.D_enc],{self.u : u_,self.isTrain : False})
            im_re, im_z, im_z_c= self.sess.run([self.re_image, self.E_u, self.E_u_c],{self.u : u_, self.isTrain : False})
            Q_c, im_re_enc= self.sess.run([self.Q_fake,self.D_enc],{self.u : im_re, self.z : im_z, self.c : im_z_c, self.isTrain : False})

            for i in range(self.batch_size) :

                feature_e = np.mean(np.sqrt((np.reshape(im_enc[0][i],(-1,100))-np.reshape(im_re_enc[i],(-1,100)))**2))
                feature_cos = 1 - cosine_similarity(np.reshape(im_enc[0][i],(-1,100)),np.reshape(im_re_enc[i],(-1,100)))
                sign_e = np.mean(np.abs(np.sign(im_enc)-np.sign(im_re_enc)))
                residual_e = np.mean(np.sqrt((np.reshape(u_[i],(1,64*64))-np.reshape(im_re[i],(1,64*64)))**2))
                residual_cos = 1 - cosine_similarity(np.reshape(u_[i],(1,64*64)),np.reshape(im_re[i],(1,64*64)))

                test_normal_feature_mse.append(feature_e)
                test_normal_feature_cosine.append(feature_cos)
                test_normal_feature_signcode.append(sign_e)
                test_normal_residual_mse.append(residual_e)
                test_normal_residual_cosine.append(residual_cos)
                
        for iteration in range(self.data.test_anomalous_data.shape[0] // self.batch_size) : 
            test_images = self.data.test_anomalous_data[iteration*self.batch_size : (iteration+1)*self.batch_size]      
            u_ = np.reshape(test_images,(-1,64,64,1)) 

            im_enc= self.sess.run([self.D_enc],{self.u : u_,self.isTrain : False})
            im_re, im_z, im_z_c= self.sess.run([self.re_image, self.E_u, self.E_u_c],{self.u : u_, self.isTrain : False})
            Q_c, im_re_enc= self.sess.run([self.Q_fake,self.D_enc],{self.u : im_re, self.z : im_z, self.c : im_z_c, self.isTrain : False})

            for i in range(self.batch_size) :

                feature_e = np.mean(np.sqrt((np.reshape(im_enc[0][i],(-1,100))-np.reshape(im_re_enc[i],(-1,100)))**2))
                feature_cos = 1 - cosine_similarity(np.reshape(im_enc[0][i],(-1,100)),np.reshape(im_re_enc[i],(-1,100)))
                sign_e = np.mean(np.abs(np.sign(im_enc)-np.sign(im_re_enc)))
                residual_e = np.mean(np.sqrt((np.reshape(u_[i],(1,64*64))-np.reshape(im_re[i],(1,64*64)))**2))
                residual_cos = 1 - cosine_similarity(np.reshape(u_[i],(1,64*64)),np.reshape(im_re[i],(1,64*64)))

                test_anomalous_feature_mse.append(feature_e)
                test_anomalous_feature_cosine.append(feature_cos)
                test_anomalous_feature_signcode.append(sign_e)
                test_anomalous_residual_mse.append(residual_e)
                test_anomalous_residual_cosine.append(residual_cos)
                
        for iteration in range(self.data.train_normal_data.shape[0] // self.batch_size) : 
            test_images = self.data.train_normal_data[iteration*self.batch_size : (iteration+1)*self.batch_size]      
            u_ = np.reshape(test_images,(-1,64,64,1)) 

            im_enc= self.sess.run([self.D_enc],{self.u : u_,self.isTrain : False})
            im_re, im_z, im_z_c= self.sess.run([self.re_image, self.E_u, self.E_u_c],{self.u : u_, self.isTrain : False})
            Q_c, im_re_enc= self.sess.run([self.Q_fake,self.D_enc],{self.u : im_re, self.z : im_z, self.c : im_z_c, self.isTrain : False})

            for i in range(self.batch_size) :

                feature_e = np.mean(np.sqrt((np.reshape(im_enc[0][i],(-1,100))-np.reshape(im_re_enc[i],(-1,100)))**2))
                feature_cos = 1 - cosine_similarity(np.reshape(im_enc[0][i],(-1,100)),np.reshape(im_re_enc[i],(-1,100)))
                sign_e = np.mean(np.abs(np.sign(im_enc)-np.sign(im_re_enc)))
                residual_e = np.mean(np.sqrt((np.reshape(u_[i],(1,64*64))-np.reshape(im_re[i],(1,64*64)))**2))
                residual_cos = 1 - cosine_similarity(np.reshape(u_[i],(1,64*64)),np.reshape(im_re[i],(1,64*64)))

                train_normal_feature_mse.append(feature_e)
                train_normal_feature_cosine.append(feature_cos)
                train_normal_feature_signcode.append(sign_e)
                train_normal_residual_mse.append(residual_e)
                train_normal_residual_cosine.append(residual_cos)
                
                
        my_hist(test_normal_feature_mse, test_anomalous_feature_mse, train_normal_feature_mse, 
                path = self.path+'/hist_feature_mse', name = 'feature_mse')
        my_hist(test_normal_feature_cosine, test_anomalous_feature_cosine, train_normal_feature_cosine, 
                path = self.path+'/hist_feature_cosine', name = 'feature_cosine')
        my_hist(test_normal_feature_signcode, test_anomalous_feature_signcode, train_normal_feature_signcode ,
                path = self.path+'/hist_feature_signcode', name = 'feature_signcode')
        my_hist(test_normal_residual_mse, test_anomalous_residual_mse, train_normal_residual_mse,
                path = self.path+'/hist_residual_mse', name = 'residual_mse')
        my_hist(test_normal_residual_cosine, test_anomalous_residual_cosine, train_normal_residual_cosine,
                path = self.path+'/hist_residual_cosine', name = 'residual_cosine')
        
        my_roc_curve(test_normal_feature_mse, test_anomalous_feature_mse, 
                path = self.path+'/roc_feature_mse', name = 'feature_mse')
        my_roc_curve(test_normal_feature_cosine, test_anomalous_feature_cosine, 
                path = self.path+'/roc_feature_cosine', name = 'feature_cosine')
        my_roc_curve(test_normal_feature_signcode, test_anomalous_feature_signcode, 
                path = self.path+'/roc_feature_signcode', name = 'feature_signcode')
        my_roc_curve(test_normal_residual_mse, test_anomalous_residual_mse,
                path = self.path+'/roc_residual_mse', name = 'residual_mse')
        my_roc_curve(test_normal_residual_cosine, test_anomalous_residual_cosine,
                path = self.path+'/roc_residual_cosine', name = 'residual_cosine')
               
       
        with open(self.path+"/test_normal_feature_mse.txt", "w") as output:
            output.write(str(test_normal_feature_mse))       
        with open(self.path+"/ test_normal_feature_cosine.txt", "w") as output:
            output.write(str( test_normal_feature_cosine))       
        with open(self.path+"/test_normal_feature_signcode.txt", "w") as output:
            output.write(str(test_normal_feature_signcode))
        with open(self.path+"/test_normal_residual_mse.txt", "w") as output:
            output.write(str(test_normal_residual_mse))     
        with open(self.path+"/test_normal_residual_cosine.txt", "w") as output:
            output.write(str(test_normal_residual_cosine))
        
        
        with open(self.path+"/test_anomalous_feature_mse.txt", "w") as output:
            output.write(str(test_anomalous_feature_mse))        
        with open(self.path+"/test_anomalous_feature_cosine.txt", "w") as output:
            output.write(str(test_anomalous_feature_cosine))        
        with open(self.path+"/test_anomalous_feature_signcodee.txt", "w") as output:
            output.write(str(test_anomalous_feature_signcode))        
        with open(self.path+"/test_anomalous_residual_mse.txt", "w") as output:
            output.write(str(test_anomalous_residual_mse ))       
        with open(self.path+"/test_anomalous_residual_cosine.txt", "w") as output:
            output.write(str(test_anomalous_residual_cosine))
        
        
        with open(self.path+"/train_normal_feature_mse.txt", "w") as output:
            output.write(str(train_normal_feature_mse))        
        with open(self.path+"/train_normal_feature_cosine.txt", "w") as output:
            output.write(str(train_normal_feature_cosine))       
        with open(self.path+"/train_normal_feature_signcode.txt", "w") as output:
            output.write(str(train_normal_feature_signcode))       
        with open(self.path+"/train_normal_residual_mse.txt", "w") as output:
            output.write(str(train_normal_residual_mse))  
        with open(self.path+"/train_normal_residual_cosine.txt", "w") as output:
            output.write(str(train_normal_residual_cosine))
        
        
        
        




