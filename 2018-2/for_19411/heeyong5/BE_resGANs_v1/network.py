import tensorflow as tf

def up_sample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x
def down_sample(x):
    """Downsamples the image by a factor of 2."""
    xd = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return xd

def up_block(x, ch, name, isTrain = True ) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope(name):
       
        x_0 = x
        
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))  
        x = tf.layers.conv2d(x, ch,[3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))       
        x = tf.layers.conv2d(x, ch,[3,3], strides=(1,1),padding = 'same',  kernel_initializer=w_init, bias_initializer=b_init) 
        x = up_sample(x)       
            
        x_0 = up_sample(x_0)
        x_0 = tf.layers.conv2d(x_0, ch,[1,1], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 

    return x_0 + x

def down_block(x, ch, name, isTrain = True ) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope(name):
       
        x_0 = x
        
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))    
        x = tf.layers.conv2d(x, ch,[3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))       
        x = tf.layers.conv2d(x, ch,[3,3], strides=(1,1),padding = 'same',  kernel_initializer=w_init, bias_initializer=b_init) 
        x = down_sample(x)     
        
        x_0 = down_sample(x_0)
        x_0 = tf.layers.conv2d(x_0, ch,[1,1], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 

    return x_0 + x
def G(x,isTrain = True, reuse = False, name = 'G', z_size = 100) : #input = (minibatch * w * h * ch)
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('G',reuse=reuse)  :
        
 
        
        fc0  = tf.reshape(x, (-1, z_size))
        w1 = tf.get_variable('w1',[z_size, 4*4*16*n],initializer=w_init)
        b1 = tf.get_variable('b1',[4*4*16*n],initializer=b_init)                                         
        fc1 = tf.matmul(fc0,w1) + b1
        
        r0  = tf.reshape(fc1, (-1, 4,4,16*n))   #4*4*512(16*n)
        
        r1 = up_block(r0, 8*n, 'block_1', isTrain) #8*8*256(8*n)
        r2 = up_block(r1, 4*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = up_block(r2, 2*n, 'block_3', isTrain) #32*32*64(2*n)
        r4 = up_block(r3, n, 'block_4', isTrain) #64*64*32(n)
        
        
        r5 = tf.nn.elu(tf.layers.batch_normalization(r4,training=isTrain))#32*32*64

        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'same',
                kernel_initializer=w_init, bias_initializer=b_init) #64*64*1
        
    r6= tf.nn.tanh(conv6,name=name)#64*64*1
  
    return r6

def E(x, isTrain = True, reuse = False, name = 'E', z_size = 100) : #input = (minibatch * w * h * ch)
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('E',reuse=reuse)  :

        conv1 = tf.layers.conv2d(x,n, [3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) #64*64*32
        r1 = down_block(conv1, 2*n, 'block_1', isTrain) #32*32*64(2*n)
        r2 = down_block(r1, 4*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = down_block(r2, 8*n, 'block_3', isTrain) #8*8*256(8*n)
        r4 = down_block(r3, 16*n, 'block_4', isTrain) #4*4*512(16*n)
        conv5 = tf.layers.conv2d(r4,z_size,[4,4], strides=(1,1),padding = 'valid', kernel_initializer=w_init, bias_initializer=b_init) #1*1*100
        
            
    r6 = tf.nn.tanh(tf.layers.batch_normalization(conv5,training=isTrain), name = name)#4*4*512
  
  
    return r6


def D_enc(x,isTrain=True,reuse = False, name = 'D_enc', feature_size = 100) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('D_enc', reuse=reuse) :
        conv1 = tf.layers.conv2d(x,n, [3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) #64*64*32
       
        r1 = down_block(conv1, 2*n, 'block_1', isTrain) #32*32*64(2*n)
        r2 = down_block(r1, 4*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = down_block(r2, 8*n, 'block_3', isTrain) #8*8*256(8*n)
        r4 = down_block(r3, 16*n, 'block_4', isTrain) #4*4*512(16*n)
        
        conv5 = tf.layers.conv2d(r4,feature_size,[4,4], strides=(1,1),padding = 'valid', kernel_initializer=w_init, bias_initializer=b_init) #1*1*100
        

        r5 = tf.layers.batch_normalization(conv5,training=isTrain)
    
    return tf.add(r5,0.0,name=name)

def D_dec(x,isTrain=True,reuse = False, name = 'D_dec') :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('D_dec', reuse=reuse) :
        
        conv1 = tf.layers.conv2d_transpose(x,16*n,[4,4], strides=(1,1),padding = 'valid',  kernel_initializer=w_init, bias_initializer=b_init)
      
        r1 = up_block(conv1, 8*n, 'block_1', isTrain) #8*8*256(8*n)
        r2 = up_block(r1, 4*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = up_block(r2, 2*n, 'block_3', isTrain) #32*32*64(2*n)
        r4 = up_block(r3, n, 'block_4', isTrain) #64*64*32(n)        
        r5 = tf.nn.elu(tf.layers.batch_normalization(r4,training=isTrain))#32*32*64

        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) #64*64*1
        
    r6= tf.nn.tanh(conv6,name=name)#64*64*1

    return r6


    

