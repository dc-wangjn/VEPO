import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# state_ratio_model = VisitationRatioModel(model, optimizer, replay_buffer,
#                  target_policy, behavior_policy, medians=None)
# state_ratio_model.fit(batch_size=32, gamma=0.99, max_iter=100)
# state_ratio = state_ratio_model.predict(state; S, A)

""" TODO
算法上就是equation (B.1)里的第二项不是(1-gamma) f(S_{i,t+1},S_{i,t},A_{i,t})了，而是(1-gamma) f(S_{i,t},S_{i,t},A_{i,t})

相当于最后那个式子，D里面的t_1+1,t_2+1都要改成t_1和t_2, 而t_1'+1,t_2'+1保持不变
"""

class VisitationRatioModel_init_SA:
    """ w(S; S, A) <- model
    """
    def __init__(self, replay_buffer,
                 target_policy, behavior_policy
                 , S_dims, h_dims, A_dims = 1
                 , optimizer = None, medians=None):
        self.model = StateDensityRatioModel_init_SA(S_dims, h_dims, A_dims)
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.medians = medians
        self.losses = []
        if not optimizer:
            lr = 0.0005
            w_clipping_val = 0.5
            w_clipping_norm = 1.0
            self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
        else:
            self.optimizer = optimizer

    def _compute_medians(self, n=100):
        transitions = self.replay_buffer.sample(n)
        states, actions, next_states = transitions[0], transitions[1], transitions[3]
        mat = tf.concat([states, actions[:,np.newaxis], next_states], axis=-1)  # n x ...
        dxx = tf.repeat(mat, n, axis=0) - tf.tile(mat, [n, 1])                  # n2 x ...
        medians = np.median(tf.math.abs(dxx), axis=0) + 1e-2 # p
        return medians
    
    def _normalize(self, weights, batch_size):
        weights = tf.reshape(weights, [batch_size, batch_size])
        weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
        weights = weights / weights_sum
        return tf.reshape(weights, [batch_size**2])
        
    def _compute_loss(self, states, actions, next_states, gamma):
        """ 
        get_A_prob(states, actions = None)
        get_greedy_A(states)
        sample_A(states)
        """
        batch_size = states.shape[0]

        states_r = tf.repeat(states, batch_size, axis=0)      # n2 x ...
        actions_r = tf.repeat(actions, batch_size)            # n2 
        actions_r = tf.expand_dims(actions_r, 1)
        states_t = tf.tile(states, [batch_size, 1])           # n2 x ...
        next_states_t = tf.tile(next_states, [batch_size, 1]) # n2 x ...
        
        ### state visitation ratios & policy ratios & deltas
        """S, gene_S, A"""
        weights = self.model.predict(tf.concat([states_r, states_t, actions_r], axis=-1))           # n2
        next_weights = self.model.predict(tf.concat([states_r, next_states_t, actions_r], axis=-1)) # n2
        weights = self._normalize(weights, batch_size)                # n2
        next_weights = self._normalize(next_weights, batch_size)      # n2 
        policy_ratios = self.target_policy.get_A_prob(states = states, actions = actions, to_numpy = True) / (self.behavior_policy.get_A_prob(states, actions)+1e-3) # n
        policy_ratios = tf.tile(policy_ratios, [batch_size])          # n2
        policy_ratios = tf.cast(policy_ratios, weights.dtype)
        deltas = gamma * weights * policy_ratios - next_weights       # n2
        
        ### kernels
        actions_r = tf.cast(actions_r, states.dtype)
        mat1 = tf.concat([states, actions[:,None], next_states], axis=-1)        # n x ...
        mat2 = tf.concat([states_r, actions_r, next_states_t], axis=-1)  # n2 x ...
        """ Laplacian Kernel
        pairwise difference: Delta
        """
        dxx1 = tf.repeat(mat2, batch_size**2, axis=0) - tf.tile(mat2, [batch_size**2, 1]) # n4 x ...
        dxx2 = tf.repeat(mat1, batch_size**2, axis=0) - tf.tile(mat2, [batch_size, 1])    # n3 x ...
        dxx3 = tf.repeat(mat1, batch_size, axis=0)    - tf.tile(mat1, [batch_size, 1])    # n2 x ...
        dxx1 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx1)/self.medians, axis=-1)) # n4
        dxx2 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx2)/self.medians, axis=-1)) # n3
        dxx3 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx3)/self.medians, axis=-1)) # n2
        
        ### final loss
        loss = tf.reduce_sum( tf.repeat(deltas, batch_size**2) * tf.tile(deltas, [batch_size**2]) * dxx1 )/batch_size**4 + \
               2*(1-gamma)*tf.reduce_sum( tf.tile(deltas, [batch_size]) * dxx2 )/batch_size**3 + \
               (1-gamma)**2*tf.reduce_sum(dxx3)/batch_size**2
        
        return loss
        
    def fit(self, batch_size=32, gamma=0.99, max_iter=100):
        if self.medians is None:
            self.medians = self._compute_medians()
            
        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            states, actions, next_states = transitions[0], transitions[1], transitions[3]
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                loss = self._compute_loss(states, actions, next_states, gamma)
            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            self.losses.append(loss.numpy())
    
    def predict(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        if inputs.shape[0] > batch_size:
            n_batch = inputs.shape[0] // batch_size + 1
            input_batches = np.array_split(inputs, n_batch)
            return np.vstack([self.model.predict(inputs).cpu().numpy() for inputs in input_batches])
        else:
            return self.model.predict(inputs).cpu().numpy() 
##############################################################################################################################

class StateDensityRatioModel_init_SA(tf.keras.Model):
    ''' weights = self.model(states_r, actions_r, states_t)
    initial? a random different?
    
    Input for `VisitationRatioModel()`; NN parameterization
    
    incoporated into the above class.
    '''
    def __init__(self, S_dims, h_dims, A_dims = 1):
        super(StateDensityRatioModel_init_SA, self).__init__()
        self.hidden_dims = h_dims

        self.input_dim = S_dims + A_dims + S_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]

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

    def predict(self, inputs):
        """
        inputs are concatenations of S, A, S_t = [r,r,t]
        where do we build the model?
        parameter for the depth!!!
        """
        z = tf.cast(inputs, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        # out = tf.matmul(h2, self.w3) + self.b3
        out = tf.matmul(h1, self.w3) + self.b3
        return out
