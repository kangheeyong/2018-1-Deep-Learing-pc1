import tensorflow as tf



def up_block(x, ch, name, isTrain = True ) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope(name):
        x0 = x
        x = tf.layers.conv2d(x, 2*ch,[3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))   
        x = x + x0
        x = tf.layers.conv2d_transpose(x, ch,[3,3], strides=(2,2),padding = 'same',  kernel_initializer=w_init, bias_initializer=b_init)  
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))          

    return x

def down_block(x, ch, name, isTrain = True ) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    
    with tf.variable_scope(name):
       
        x = tf.layers.conv2d(x, ch,[3,3], strides=(2,2),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) 
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))    
        x0 = x
        x = tf.layers.conv2d(x, ch,[3,3], strides=(1,1),padding = 'same',  kernel_initializer=w_init, bias_initializer=b_init) 
        x = tf.nn.elu(tf.layers.batch_normalization(x,training=isTrain))       
        x = x + x0
    return x
def G(x,isTrain = True, reuse = False, name = 'G', z_size = 100) : #input = (minibatch * w * h * ch)
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('G',reuse=reuse)  :
        
        conv0 = tf.layers.conv2d_transpose(x,32*n,[4,4], strides=(1,1),padding = 'valid',  kernel_initializer=w_init, bias_initializer=b_init)
        r0 = tf.nn.elu(tf.layers.batch_normalization(conv0,training=isTrain))   
               
        r1 = up_block(r0, 16*n, 'block_1', isTrain) #8*8*256(8*n)
        r2 = up_block(r1, 8*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = up_block(r2, 4*n, 'block_3', isTrain) #32*32*64(2*n)
        r4 = up_block(r3, 2*n, 'block_4', isTrain) #64*64*32(n)
        r5 = up_block(r4, n, 'block_5', isTrain) #64*64*32(n)
        
        conv6 = tf.layers.conv2d(r5,1,[3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) #64*64*1

    return  tf.nn.sigmoid(conv6,name=name)



def D(x,isTrain=True,reuse = False, name = 'D', feature_size = 100) :
    
    w_init = tf.truncated_normal_initializer(mean= 0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    n = 16
    with tf.variable_scope('D', reuse=reuse) :
        conv0 = tf.layers.conv2d(x,n, [3,3], strides=(1,1),padding = 'same', kernel_initializer=w_init, bias_initializer=b_init) #64*64*32
        r0 = tf.nn.elu(conv0) 
        
        r1 = down_block(r0, 2*n, 'block_1', isTrain) #32*32*64(2*n)
        r2 = down_block(r1, 4*n, 'block_2', isTrain) #16*16*128(4*n)
        r3 = down_block(r2, 8*n, 'block_3', isTrain) #8*8*256(8*n)
        r4 = down_block(r3, 16*n, 'block_4', isTrain) #4*4*512(16*n)
        r5 = down_block(r4, 32*n, 'block_5', isTrain) #4*4*512(16*n)
        
        conv6 = tf.layers.conv2d(r5,1,[4,4], strides=(1,1),padding = 'valid', kernel_initializer=w_init, bias_initializer=b_init) #1*1*100
        r6 = tf.layers.batch_normalization(conv6,training=isTrain)
    
    return tf.nn.sigmoid(r6,name=name)


    


    

