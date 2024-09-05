import tensorflow.compat.v1 as tf 
import numpy as np
import time


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

class neural_net(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
            
    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y

def Navier_Stokes_2D(c, u, v, p, t, x, y, Pec, Rey):
    
    Y = tf.concat([c, u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Y[:,3:4]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
    
    return e1, e2, e3, e4

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]

def Navier_Stokes_3D(c, u, v, w, p, t, x, y, z, Pec, Rey):
    
    Y = tf.concat([c, u, v, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]
    
    e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e1, e2, e3, e4, e5

def steady_state_Navier_Stokes_3D_R(c, u, v, w, p, x, y, z, R, Pec, Rey):
    
    Y = tf.concat([c, u, v, w, p], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_R = fwd_gradients(Y, R)


    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    Y_RR = fwd_gradients(Y_R, R)    
    
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]
    
    e1 = (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
        
    return e1, e2, e3, e4, e5

def steady_state_Navier_Stokes_3D(c, u, v, w, p, x, y, z,  Pec, Rey):
    
    Y = tf.concat([c, u, v, w, p], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]
    
    e1 = (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
        
    return e1, e2, e3, e4, e5


def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = tf.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]


class model(object):
    
    def __init__(self, x_data, y_data, z_data, c_data, 
                       x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.x_data, self.y_data, self.z_data, self.c_data] = [x_data, y_data, z_data,  c_data] 
        [self.x_eqns, self.y_eqns, self.z_eqns] = [x_eqns, y_eqns, z_eqns]
        
        # placeholders
        [self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        
        # physics "uninformed" neural networks
        self.net_cuvwp = neural_net(self.x_data, self.y_data, self.z_data, layers=self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)
         
        # physics "informed" neural networks
        
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred,
         self.e5_eqns_pred] = steady_state_Navier_Stokes_3D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.w_eqns_pred,
                                               self.p_eqns_pred,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.z_eqns_tf,
                                               self.Pec,
                                               self.Rey)
        
        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0) 
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    #def train(self, total_time, learning_rate):
    def train(self, Niter, learning_rate):

        
        loss_history = []        
        
        N_data = self.x_data.shape[0]
        N_eqns = self.x_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        
        #while running_time < total_time:
        while it < Niter: 
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (x_data_batch,
             y_data_batch,
             z_data_batch,            
             c_data_batch) = (self.x_data[idx_data, :],
                              self.y_data[idx_data, :],
                              self.z_data[idx_data, :],
                              self.c_data[idx_data, :])

            (x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.x_eqns[idx_eqns, :],
                              self.y_eqns[idx_eqns, :],
                              self.z_eqns[idx_eqns, :])

            tf_dict = {self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,                     
                       self.c_data_tf: c_data_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed 
                [loss_value,
                 c_data_pred,
                 c_data_tf,
                 e1_eqns_pred_value,
                 e2_eqns_pred_value,
                 e3_eqns_pred_value,
                 e4_eqns_pred_value,
                 e5_eqns_pred_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.c_data_pred,
                                                       self.c_data_tf,
                                                       self.e1_eqns_pred,
                                                       self.e2_eqns_pred,
                                                       self.e3_eqns_pred,
                                                       self.e4_eqns_pred,
                                                       self.e5_eqns_pred,
                                                       self.learning_rate], tf_dict)   
                                                       
                                                       
                loss_history.append(loss_value)
                                
                print('It: %d, Loss: %.3e, Running Time: %.2fs, Learning Rate: %.1e' % (it, loss_value, running_time, learning_rate_value))
                start_time = time.time()
            it += 1
            
            
        return loss_history
            

    
    def predict(self, x_star, y_star, z_star):
        
        tf_dict = {self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, w_star, p_star    
