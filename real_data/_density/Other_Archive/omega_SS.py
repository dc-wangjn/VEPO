import numpy as np
import tensorflow as tf

# state_ratio_model = VisitationRatioModel(model, optimizer, replay_buffer,
#                  target_policy, behavior_policy, medians=None)
# state_ratio_model.fit(batch_size=32, gamma=0.99, max_iter=100)
# state_ratio = state_ratio_model.predict(state; S)

""" TODO
算法上就是equation (B.1)里的第二项不是(1-gamma) f(S_{i,t+1},S_{i,t},A_{i,t})了，而是(1-gamma) f(S_{i,t},S_{i,t},A_{i,t})

相当于最后那个式子，D里面的t_1+1,t_2+1都要改成t_1和t_2, 而t_1'+1,t_2'+1保持不变
"""

class VisitationRatioModel_init_S:
    """ w(S; S) <- model
    """
    def __init__(self, replay_buffer, S_dims, h_dims, gamma
                 , target_policy, behavior_policy
                 , optimizer = None, medians=None):
        self.model = StateDensityRatioModel_init_S(S_dims, h_dims)
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.medians = medians
        self.losses = []
        self.gamma = gamma
        if not optimizer:
            lr = 0.0005
            w_clipping_val = 0.5
            w_clipping_norm = 1.0
            self.optimizer = tf.keras.optimizers.Adam(lr = lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
        else:
            self.optimizer = optimizer
            
    ############################################################################################################
    def _compute_medians(self, n=100):
        transitions = self.replay_buffer.sample(n)
        states, next_states = transitions[0], transitions[3]
        mat = tf.concat([states, next_states], axis=-1)  
        dxx = tf.repeat(mat, n, axis=0) - tf.tile(mat, [n, 1])
        medians = np.median(tf.math.abs(dxx), axis=0) + 1e-4 # p
        return medians
    
    def _normalize(self, weights, batch_size):
        """ M^* = M """
        weights = tf.reshape(weights, [batch_size, batch_size])
        weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
        weights = weights / weights_sum
        return tf.reshape(weights, [batch_size**2])

    ############################################################################################################
    # repeat([a,b], 2) - tile([a,b], 2) = [a,a,b,b] - [a,b,a,b] = [a-a, a-b, b-a, b-b]
    # [a,a,a,b,b,b,c,c,c] - [a,b,c,a,b,c,a,b,c] = [a-a, a-b, a-c, b-a, b-b, b-c, c-a, c-b, c-c] 
    # -> target: pairwise; very confusing
    #####
    def _compute_loss(self, states, actions, next_states):
        """
        predict input = [r, t]
        overlap terms -> zero-value for K(, )
        median: why distance between SAS and SAS?
        TODO: the loss normalization term -> final prediciton output
        (M, 2) term
        Runzhe added one more K() term for each
        # be careful and 反了?
        # the SA file correct?
        (1)
        current: rr[r,t] * tt[r,t] * [rr[r,t] - tt[r,t]]
        target: 11' * 22' * 1'2' * 12 -> 1'2' = rrt ttt; 12 = rrr ttr
        (2)
        current: t[r,t] *  (rr[] - t[r,t])
        target: 11' * 1'2 * 12 + 22' * 12' * 12
        with tr = 1, tt = 1', rt = 2, rr = 2'
        (3) should be correct with tf.square
        """
        
        batch_size, dim = states.shape
        states_r = tf.repeat(states, batch_size, axis=0)      # n2 
        states_t = tf.tile(states, [batch_size, 1])           # n2 
        next_states_t = tf.tile(next_states, [batch_size, 1]) # n2 
        
        ### (1) DELTA: state visitation ratios & policy ratios & deltas
        """ SHAPE for predict is incorrect? """
        weights = self.model.predict(tf.concat([states_r, states_t], axis=-1), batch_size)           # n2
        next_weights = self.model.predict(tf.concat([states_r, next_states_t], axis=-1), batch_size) # n2
        weights = self._normalize(weights, batch_size)                # n2
        next_weights = self._normalize(next_weights, batch_size)      # n2 
        
        policy_ratios = self.target_policy.get_A_prob(states, actions) / (self.behavior_policy.get_A_prob(states, actions)+1e-3) # n
        policy_ratios = tf.tile(policy_ratios, [batch_size])          # n2
        policy_ratios = tf.cast(policy_ratios, weights.dtype)         # n2
        deltas = self.gamma * weights * policy_ratios - next_weights       # n2 [r,t]
        
        ### (2) kernels: symmetric
        mat1 = tf.concat([states, next_states], axis=-1)  # n
        mat2 = tf.concat([states_r, next_states_t], axis=-1) # n2; [r,t]
        dxx1 = tf.repeat(mat2, batch_size**2, axis=0) - tf.tile(mat2, [batch_size**2, 1])  # n4; [rr[r,t] - tt[r,t]]
        dxx2 = tf.repeat(mat1, batch_size**2, axis=0) - tf.tile(mat2, [batch_size, 1]) # n3 rr[] - t[r, t]
        dxx3 = tf.repeat(mat1, batch_size, axis=0)    - tf.tile(mat1, [batch_size, 1]) # n2   r - t
        # Laplacian Kernel: e(- sum(|x - y|, 1))
        
        dxx_tt_rt = tf.tile(next_states_t, [batch_size, 1]) - tf.repeat(states_t, batch_size, axis=0)
        K_term1_1 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx1[dim:])/self.medians[dim:], axis=-1))
        K_term1_2 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx1[:dim])/self.medians[:dim], axis=-1))
        K_term2 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx_tt_rt)/self.medians[:dim], axis=-1))
        dxx3 = tf.exp(-tf.math.reduce_sum(tf.math.abs(dxx3)/self.medians, axis=-1)) # n2
        
        term1 = tf.reduce_sum( tf.repeat(deltas, batch_size**2) * tf.tile(deltas, [batch_size**2]) * K_term1_1 * K_term1_2 /batch_size**4 )
        term2 = 2*(1-self.gamma)*tf.reduce_sum( tf.tile(deltas, [batch_size]) * K_term2 )/batch_size**3
        term3 = (1-self.gamma)**2*tf.reduce_sum(tf.square(dxx3))/batch_size**2
        ### D(): final loss
        loss = term1 + term2 + term3 
                             
        return loss
        
    def fit(self, batch_size=32, max_iter=100):
        if self.medians is None:
            self.medians = self._compute_medians()
            
        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            states, next_states = transitions[0], transitions[3]
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                """ ??? CHECK !! carefully ! 
                the last line of page 7? confusing!
                """
                loss = self._compute_loss(states, actions, next_states, self.gamma)
            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            self.losses.append(loss.numpy())
            
##############################################################################################################################

class StateDensityRatioModel_init_S(tf.keras.Model):
    ''' weights = self.model(states_r, states_t)
    
    Input for `VisitationRatioModel_init_S()`; NN parameterization
    incoporated into the above class.
    '''
    def __init__(self, S_dims, h_dims):
        super(StateDensityRatioModel_init_S, self).__init__()
        self.hidden_dims = h_dims

        self.input_dim = S_dims + S_dims
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

    def predict(self, inputs, batch_size):
        """ input = [r, t]
        introduce a parameter for the depth (but can not be here? too weak?)
        """
        z = tf.reshape(tensor=inputs, shape=[batch_size, -1])
        z = tf.cast(z, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        # h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        # out = tf.matmul(h2, self.w3) + self.b3
        out = tf.matmul(h1, self.w3) + self.b3
        return out
