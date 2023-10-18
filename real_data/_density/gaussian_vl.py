from _density.GAN_utils import *
from scipy.linalg import sqrtm

# from _util import *
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#####################################################################################################################
class linear_gaus_transition(tf.keras.Model):

    def __init__(self, z_dims, h_dims, v_dims, x_dims, batch_size,cov,use_aitken):
        super(linear_gaus_transition, self).__init__()
        self.hidden_dims = h_dims
        self.batch_size = batch_size
        self.cov = cov
        self.use_aitken = use_aitken

        self.input_dim = z_dims + v_dims
        self.input_shapes = [self.input_dim, x_dims]

        self.w1 = self.xavier_var_creator(self.input_shapes)
        self.b1 = tf.Variable(tf.zeros(self.input_shapes[1], tf.float64))


    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        mean = tf.matmul(z, self.w1) + self.b1

        return mean,mean,mean

class gaus_transition(tf.keras.Model):

    def __init__(self, z_dims, h_dims, x_dims, batch_size,use_aitken):
        super(gaus_transition, self).__init__()
        self.hidden_dims = h_dims
        self.batch_size = batch_size
#         self.cov = cov
        self.use_aitken = use_aitken

        self.input_dim = z_dims 
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, x_dims]
        
        

        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        h1 = tf.nn.swish(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.swish(tf.matmul(h1, self.w2) + self.b2)
        mean = tf.matmul(h2,self.w3)+ self.b3
#         log_var = tf.matmul(h1, self.w3) + self.b3
#         var = tf.math.exp(log_var)
#         log_var = tf.squeeze(log_var)
#         var = tf.squeeze(var)

#         return mean, log_var, var
        return mean,mean,mean


class CondSampler():
    """ the class learned with `learn_cond_sampler()`
    S, A -> SS
    """

    def __init__(self, generator_x, non_overlap_dim):
#         self.v_dist = v_dist
#         self.v_dims = v_dims
        self.generator_x = generator_x
        self.seed = 42
        self.non_overlap_dim = non_overlap_dim
        
#         self.std = sqrtm(generator_x.cov)
#         self.use_aitken = generator_x.use_aitken

    def sample(self, S, A):
        test_z = tf.concat([S, A], axis=1)
        test_z = tf.cast(test_z, tf.float64)
#         noise_v = tf.cast(self.v_dist.sample(test_z.shape[0], self.seed), tf.float64)
#         noise_v = tf.cast(np.zeros((test_z.shape[0],self.v_dims)),tf.float64)# modified by wjn 5-6
        g_inputs = tf.cast(test_z,tf.float64)
        # generator samples from G and evaluate from D
        x_mean, x_logvar, x_var = self.generator_x.call(g_inputs, training=False)
#         x_samples = np.random.multivariate_normal(x_mean.numpy(), x_var.numpy())
        
#         if self.use_aitken == 'True':
#             rditem = np.random.randn(x_mean.numpy().shape[0],x_mean.numpy().shape[1])
#             x_samples = np.clip(0.1*np.dot(rditem,self.std) + x_mean.numpy(),0.05,0.95)# modified by wjn 5-23
#         else:
        rditem = np.random.randn(x_mean.numpy().shape[0],x_mean.numpy().shape[1])
        x_samples = np.clip(0.1*rditem + x_mean.numpy(),0.05,0.95)# modified by wjn 5-23
        
#         x_samples = np.random.randn(x_mean.shape) + x_mean
        self.seed += 1
        if self.non_overlap_dim > 0 and self.non_overlap_dim +1 < test_z.shape[1]: 
#             print(self.non_overlap_dim)
#             print(test_z.shape[1])
            return np.maximum(np.hstack([test_z[:, self.non_overlap_dim+1:], x_samples]),0)
        else:
#             print('hello!')
            return np.maximum(x_samples,0)


class v_dist_sampler():
    def __init__(self, means, stds):
        self.means = np.array(means)
        self.cov = np.diag(np.array(stds) ** 2)

    def sample(self, n, seed=42):
        np.random.seed(seed)
        return np.random.multivariate_normal(self.means, self.cov, n)


def learn_cond_sampler(trajs=None, num_A = None, epochs=100, non_overlap_dim=3
                       , batch_size=64,  v_dims=int(3), h_dims=int(3)
                       ,
                       optimizer={"lr": 0.0005, "w_clipping_norm": 1.0, "w_clipping_val": 0.5, "gen_clipping_norm": 1.0,
                                  "gen_clipping_val": 0.5}
                       , use_aitken=False):
    """ apply GAN to estimate conditional distribution and learn a generator, which generates (SS | A, S) <=> x <- z

    TODO: should keep the testing -> CV part.

    Args:
        a training dataset {(SS, A, S)}_i
    """
    #####################################################################################################################
    """ CHECK THESE HYPER-PARAMETERS """

    seed = 0

    optimiser = tf.keras.optimizers.Adam(optimizer["lr"], beta_1=0.5, clipnorm=optimizer["gen_clipping_norm"],
                                            clipvalue=optimizer["gen_clipping_val"])

    states, actions, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in [0, 1, 3]]

    z = np.hstack([states, np.expand_dims(actions/(num_A-1), axis=1)]) # modified by wjn 5-6
    x = next_states.copy()
    overlap_dim = z.shape[1] - 1 - non_overlap_dim
    if overlap_dim > 0:
        x = x[:, overlap_dim:]

    
    training = tf.data.Dataset.from_tensor_slices((x, z))
    training = training.repeat(epochs)
    batched_training = training.shuffle(300).batch(batch_size)

    z_dim = z.shape[1]
    x_dims = x.shape[1]

    #####################################################################################################################

    # no. of random and hidden dimensions
    # v_dist = tfp.distributions.Normal([0 for i in range(v_dims)], scale= [1.0 / v_dims for i in range(v_dims)])
#     v_dist = v_dist_sampler([0 for i in range(v_dims)], [1.0 / v_dims for i in range(v_dims)])

    # create instance of G & D
    generator_x = gaus_transition(z_dim, h_dims, x_dims, batch_size,use_aitken )

    @tf.function
    def x_update_g(real_x, real_z, gx_optimiser, generator_x):
        gen_inputs = tf.cast(real_z,tf.float64)
        out_put_dim = real_x.shape[0]

        with tf.GradientTape() as gen_tape:
            x_mean, x_logvar, x_var = generator_x.call(gen_inputs)#(batch,3)


#             if use_aitken == 'True':
#                 mean_part = tf.math.squared_difference(x_mean,real_x)

#                 # Mahalanobis distance instead of least square
#                 loss = tf.linalg.trace(tf.tensordot(tf.transpose(mean_part),tf.tensordot(mean_part,cov_inv,axes=1),axes=1))/out_put_dim
    #             quad = tf.math.reduce_mean(tf.math.reduce_sum(mean_part/x_var,axis=1))

    #             l1 = out_put_dim * tf.math.reduce_sum(tf.math.abs(x_logvar))
    #             loss = l1 + quad

#             else:
            mean_part = tf.math.squared_difference(x_mean,real_x)
            loss = tf.math.reduce_mean(tf.math.reduce_sum(mean_part,axis=1))

        generator_grads = gen_tape.gradient(loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))

        return loss,x_var



    #####################################################################################################################
    #####################################################################################################################
    current_iters = 0
    for x_batch, z_batch in batched_training.take(epochs):
        """ SGD updating: GAN training
        """

#         noise_v = v_dist.sample(batch_size, seed)
#         noise_v = np.zeros((batch_size,v_dims))  # modified by wjn 5-6
        #         noise_v = v_dist.sample([batch_size], seed = seed)
#         noise_v = tf.cast(noise_v, tf.float64)

        seed += 1

        loss_x,var = x_update_g(x_batch, z_batch, optimiser,generator_x)
        # loss_x = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p, gx_optimiser, discriminator_x,
        #                     generator_x)
        #         with train_writer.as_default():
        # tf.summary.scalar('Wasserstein X Discriminator Loss', x_disc_loss, step=current_iters)
#         tf.summary.scalar('Wasserstein X GEN Loss', loss_x, step=current_iters)
#         print('generator loss ->{} l1 is {}, quad is {}'.format(loss_x,l1,quad))
#         print('generator loss ->{} var is {}'.format(loss_x,var))
        if current_iters%500==0:
            print('iter {} generator loss ->{} '.format(current_iters, loss_x))
        #             train_writer.flush()
        current_iters += 1
    #####################################################################################################################
    cond_sampler = CondSampler( generator_x=generator_x, non_overlap_dim=non_overlap_dim)
    return cond_sampler


"""
n = 100
nstd=1.0

a_x=0.05 
M=500
k=2 
var_idx=1
z_dim = 20

beta = randn(z_dim)
z = randn(n, z_dim)
A = np.matmul(z, beta) + randn(n) / 10
A = A.reshape(-1, 1)
zz = randn(n, z_dim)

learn_cond_sampler(ZAZZ = [z, A, zz])
"""