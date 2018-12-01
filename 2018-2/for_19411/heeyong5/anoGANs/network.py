import tensorflow as tf



def G(x,c,isTrain = True, reuse = False, name = 'G') : #input = (minibatch * w * h * ch)
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)

    with tf.variable_scope('G',reuse=reuse)  :
        
        x_concat = tf.concat([x,c],3)
        conv1 = tf.layers.conv2d_transpose(x_concat,512,[4,4], strides=(1,1),padding = 'valid',
                kernel_initializer=w_init, bias_initializer=b_init) 
        r1 = tf.nn.elu(tf.layers.batch_normalization(conv1,training=isTrain))#4*4*512
        
        conv2 = tf.layers.conv2d_transpose(r1,256,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#8*8*256
                
        conv3 = tf.layers.conv2d_transpose(r2,128,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#16*16*128

        conv4 = tf.layers.conv2d_transpose(r3,64,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#32*32*64

        conv5 = tf.layers.conv2d_transpose(r4,1,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init) #64*64*1
        
    r5= tf.nn.tanh(conv5,name=name)#64*64*1
  
    return r5

def E(x, isTrain = True, reuse = False, name = 'E', z_size = 100, c_size = 10) : #input = (minibatch * w * h * ch)
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)

    with tf.variable_scope('E',reuse=reuse)  :

        conv1 = tf.layers.conv2d(x,64,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init) 
        r1 = tf.nn.elu(conv1)#32*32*64
        
        conv2 = tf.layers.conv2d(r1,128,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#16*16*128
                
        conv3 = tf.layers.conv2d(r2,256,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#8*8*256

        conv4 = tf.layers.conv2d(r3,512,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init)
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain))#4*4*512

        conv5 = tf.layers.conv2d(r4,z_size,[4,4], strides=(1,1),padding = 'valid',
                kernel_initializer=w_init, bias_initializer=b_init) #1*1*100
        
        #
        
        fc0  = tf.reshape(conv4, (-1, 4*4*512))
        
        w1 = tf.get_variable('w1',[4*4*512, c_size],initializer=w_init)
        b1 = tf.get_variable('b1',[c_size],initializer=b_init)                                         
        fc1 = tf.nn.softmax(tf.matmul(fc0,w1) + b1, name = name)
            
    r5 = tf.nn.tanh(tf.layers.batch_normalization(conv5,training=isTrain), name = name)#4*4*512
  
  
    return r5, tf.reshape(fc1,(-1,1,1,c_size), name = name+'_c')


def D_enc(x,isTrain=True,reuse = False, name = 'D_enc', feature_size = 100) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope('D_enc', reuse=reuse) :
       
        conv1 = tf.layers.conv2d(x,64,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init) 
        r1 = tf.nn.elu(conv1)#32*32*64

   
        conv2 = tf.layers.conv2d(r1,128,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r2 = tf.nn.elu(tf.layers.batch_normalization(conv2,training=isTrain))#16*16*128
  
        conv3 = tf.layers.conv2d(r2,256,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r3 = tf.nn.elu(tf.layers.batch_normalization(conv3,training=isTrain))#8*8*256
        
        conv4 = tf.layers.conv2d(r3,512,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)    
        r4 = tf.nn.elu(tf.layers.batch_normalization(conv4,training=isTrain), name = name)#4*4*512
        
        conv5 = tf.layers.conv2d(r4,feature_size,[4,4], strides=(1,1),padding = 'valid',
                                kernel_initializer=w_init, bias_initializer=b_init)    
        r5 = tf.layers.batch_normalization(conv5,training=isTrain)
    
    return tf.add(r5,0.0,name=name)

def D_dec(x,isTrain=True,reuse = False, name = 'D_dec') :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope('D_dec', reuse=reuse) :
        
        conv6 = tf.layers.conv2d_transpose(x,512,[4,4], strides=(1,1),padding = 'valid',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r6 = tf.nn.elu(tf.layers.batch_normalization(conv6,training=isTrain))#4*4*256
        
        conv7 = tf.layers.conv2d_transpose(r6,256,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r7 = tf.nn.elu(tf.layers.batch_normalization(conv7,training=isTrain))#8*8*256

        conv8 = tf.layers.conv2d_transpose(r7,128,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r8 = tf.nn.elu(tf.layers.batch_normalization(conv8,training=isTrain))#16*16*128
             
        conv9 = tf.layers.conv2d_transpose(r8,64,[5,5], strides=(2,2),padding = 'same',
                                kernel_initializer=w_init, bias_initializer=b_init)
        r9 = tf.nn.elu(tf.layers.batch_normalization(conv9,training=isTrain))#32*32*64
          
        conv10 = tf.layers.conv2d_transpose(r9,1,[5,5], strides=(2,2),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init) #64*64*1
        
    r10= tf.nn.tanh(conv10,name=name)#64*64*1
    
    return r10
def Q_cat(x, reuse = False, name = 'Q_cat',c_size = 10, feature_size = 100) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope('Q_cat', reuse=reuse) :

        fc0  = tf.reshape(x, (-1, feature_size))
        
        w1 = tf.get_variable('w1',[feature_size, c_size],initializer=w_init)
        b1 = tf.get_variable('b1',[c_size],initializer=b_init)                                     
        fc1 = tf.nn.softmax(tf.matmul(fc0,w1) + b1)
    
    return tf.reshape(fc1, (-1,1,1,c_size), name = name)


    

