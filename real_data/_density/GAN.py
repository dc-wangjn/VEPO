from _density.GAN_utils import *
# from _util import * 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#####################################################################################################################

class CondSampler():
    """ the class learned with `learn_cond_sampler()`
    S, A -> SS
    """
    def __init__(self, v_dist, v_dims, generator_x, non_overlap_dim,
                 minimax = False, scale_x = None, scale_z = None):
        self.v_dist = v_dist
        self.v_dims = v_dims
        self.generator_x = generator_x
        self.seed = 42
        self.non_overlap_dim = non_overlap_dim
        
        self.scale_x = scale_x
        self.scale_z = scale_z 
        
        self.minimax = minimax
        
    def sample(self, S, A): 
        """Our generateor is based on the training set S1 via rescale using the 5-th and 95-th percentile quantiles. Note as (Psz1,Plz1) for input and (Psx1,Plx1) for target.
        
    Then to generate the corresponding scale states, we should imply the same scale transformation on S2, which means the input of generator is (S2 - Psz1)/(Plz1 - Psz1), then we rescale the output by SS|S2,A -> (Plx1 - Psx1) * output + Psx1"""
        test_zo = tf.concat([S, A], axis=1)
        test_zo = tf.cast(test_zo, tf.float64)

        overlap_dim = test_zo.shape[1] - 1 - self.non_overlap_dim

        if self.minimax:
            zmax = tf.constant(self.scale_z[1])
            zmin = tf.constant(self.scale_z[0])
            test_z = (test_zo - zmin)/(zmax - zmin + 0.00001)
        #             print('test_z shape is {}'.format(test_z.shape))
        else:
            test_z = test_zo
        noise_v = tf.cast(self.v_dist.sample(test_z.shape[0], self.seed), tf.float64)
        g_inputs = tf.concat([test_z, noise_v], axis=1)
        #         print('g_input {}'.format(g_inputs))
        # generator samples from G and evaluate from D
        x_samples = self.generator_x.call(g_inputs, training=False)
        if self.minimax:
        #             print('xmax {}'.format(xmax))
        #             print('xmin {}'.format(xmin))
        #             print('x_samples {}'.format(x_samples.numpy()))
            xmax = tf.constant(self.scale_x[1])[overlap_dim:]
            xmin = tf.constant(self.scale_x[0])[overlap_dim:]
            x_samples = x_samples * (xmax - xmin) + xmin


        self.seed += 1
        #         print(test_zo[:, self.non_overlap_dim+1:].shape)

        if self.non_overlap_dim > 0:
            return np.hstack([test_zo[:, self.non_overlap_dim+1:], x_samples.numpy()])
        else:
            return x_samples.numpy()
    
class v_dist_sampler():
    def __init__(self, means, stds):
        self.means = np.array(means)
        self.cov = np.diag(np.array(stds) ** 2)
    def sample(self, n, seed = 42):
        np.random.seed(seed)
        return np.random.multivariate_normal(self.means, self.cov, n)
        
def learn_cond_sampler(trajs = None, epochs = 100, non_overlap_dim = 3
                , batch_size = 64, train_writer=None
                , v_dims = int(3), h_dims = int(3)
                , optimizer = {"lr" : 0.0005, "w_clipping_norm" : 1.0, "w_clipping_val" : 0.5, "gen_clipping_norm": 1.0, "gen_clipping_val" : 0.5}
                ,minimax = True, quantile = 95):
    """ apply GAN to estimate conditional distribution and learn a generator, which generates (SS | A, S) <=> x <- z
    
    TODO: should keep the testing -> CV part.
    
    Args:
        a training dataset {(SS, A, S)}_i
    """
    #####################################################################################################################
    """ CHECK THESE HYPER-PARAMETERS """
    scaling_coef = 1.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30
    seed = 0
    dx_optimiser = tf.keras.optimizers.Adam(optimizer["lr"], beta_1=0.5, clipnorm=optimizer["w_clipping_norm"], clipvalue=optimizer["w_clipping_val"])
    gx_optimiser = tf.keras.optimizers.Adam(optimizer["lr"], beta_1=0.5, clipnorm=optimizer["gen_clipping_norm"], clipvalue=optimizer["gen_clipping_val"])

    #####################################################################################################################
    ################################# Prepare Data #############################################################
    #####################################################################################################################
    states, actions, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in [0, 1, 3]]
    
    z = np.hstack([states, np.expand_dims(actions, axis=1)])
    x = next_states.copy()
    
    if minimax:
        z ,scale_z = rescale_with_quantile(z, quantile = quantile)
        x, scale_x = rescale_with_quantile(x, quantile = quantile)
        
    overlap_dim = z.shape[1] - 1 - non_overlap_dim
    if overlap_dim > 0:
        x = x[:, overlap_dim:]
    
    training = tf.data.Dataset.from_tensor_slices((x, z))
    training = training.repeat(epochs)
    batched_training = training.shuffle(300).batch(batch_size * 2)
    
    z_dim = z.shape[1]
    x_dims = x.shape[1]
   
    
    #####################################################################################################################
    
    # no. of random and hidden dimensions
    # v_dist = tfp.distributions.Normal([0 for i in range(v_dims)], scale= [1.0 / v_dims for i in range(v_dims)])
    v_dist = v_dist_sampler([0 for i in range(v_dims)], [1.0 / v_dims for i in range(v_dims)])
    
    # create instance of G & D
#     print('x dim is {}'.format(x_dims))
    generator_x = WGanGenerator(z_dim, h_dims, v_dims, x_dims, batch_size)
    discriminator_x = WGanDiscriminator(z_dim, h_dims, x_dims, batch_size)
    
    #####################################################################################################################
    #####################################################################################################################
    @tf.function
    def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p, dx_optimiser, discriminator_x, generator_x):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_x.call(gen_inputs)
        fake_x_p = generator_x.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
        """ generating fake with the noise
        """
        with tf.GradientTape() as disc_tape:
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                             f_real_p, f_fake_p)
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
        dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))

    @tf.function
    def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p, gx_optimiser, discriminator_x, generator_x):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_x.call(gen_inputs)
            fake_x_p = generator_x.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
        return gen_loss

    #####################################################################################################################
    #####################################################################################################################
    current_iters=0
    for x_batch, z_batch in batched_training.take(epochs):
        """ SGD updating: GAN training
        """
        if x_batch.shape[0] != batch_size * 2:
            continue
        x_batch1 = tf.cast(x_batch[0:batch_size, ...],tf.float64)
        x_batch2 = tf.cast(x_batch[batch_size:, ...],tf.float64)
        z_batch1 = tf.cast(z_batch[0:batch_size, ...],tf.float64)
        z_batch2 = tf.cast(z_batch[batch_size:, ...],tf.float64)
        
        
        noise_v = v_dist.sample(batch_size, seed)
#         noise_v = v_dist.sample([batch_size], seed = seed)
        noise_v = tf.cast(noise_v, tf.float64)
        noise_v_p = v_dist.sample(batch_size, seed)
#         noise_v_p = v_dist.sample([batch_size], seed = seed)
        noise_v_p = tf.cast(noise_v_p, tf.float64)
        seed += 1
        
        x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p, dx_optimiser, discriminator_x, generator_x)
        loss_x = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p, gx_optimiser, discriminator_x, generator_x)
    #         with train_writer.as_default():
        # tf.summary.scalar('Wasserstein X Discriminator Loss', x_disc_loss, step=current_iters)
        tf.summary.scalar('Wasserstein X GEN Loss', loss_x, step=current_iters)
        if current_iters%500==0:
            
            print('iters -> {}, GAN generator loss ->{}'.format(current_iters,loss_x))
    #             train_writer.flush()
        current_iters += 1
    #####################################################################################################################
    cond_sampler = CondSampler(v_dist = v_dist, v_dims = v_dims, generator_x = generator_x, non_overlap_dim = non_overlap_dim, minimax = minimax, scale_x = scale_x, scale_z = scale_z)
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