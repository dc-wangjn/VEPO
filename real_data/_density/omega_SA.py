import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.backend.set_floatx('float64')
# state_ratio_model = VisitationRatioModel(model, optimizer, replay_buffer,
#                  target_policy, behavior_policy, medians=None)
# state_ratio_model.fit(BS=32, gamma=0.99, max_iter=100)
# state_ratio = state_ratio_model.predict(state; S, A)


"""
1. those terms including the replay_buffer needs to be provided!
2. the corresponding estimator should be updated to the new ratio estimator!
"""

class VisitationRatioModel_init_SA():
    """ w(S, A) <- model
    see the derivations on page 2, the supplement of VE. And derivations of Lemma 17, the minimax paper. Some errors? 
    No open source. Need to be implemented ourself.
    """
    def __init__(self, replay_buffer
                 , target_policy, A_range
                 , h_dims
                 , lr = 0.0005
                 , A_dims = 1, gpu_number = 0
                 , w_clipping_val = 0.5, w_clipping_norm = 1.0, beta_1 = 0.5
                 , optimizer = None, medians=None):
        self.model = Omega_SA_Model(replay_buffer.S_dims, h_dims, A_dims, gpu_number)
        self.S_dims = replay_buffer.S_dims
        self.A_dims = A_dims
        self.gpu_number = gpu_number
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.A_range = A_range
        self.losses = []

        self.sepe_A = 0
        
        
        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1 = beta_1, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    def _compute_medians(self, n= 32, rep = 20):
        # to save memory, we do it iteratively
        # medians_11_22, medians_12_33, medians_12_34 = [tf.zeros([(self.S_dims + self.A_dims)], tf.float64) for _ in range(3)]
        with tf.device('/gpu:' + str(self.gpu_number)):
            
            if self.sepe_A:
                median = tf.zeros([(self.S_dims)], tf.float64)
            else:
                median = tf.zeros([(self.S_dims + self.A_dims)], tf.float64)
            # median = 0

            for i in range(rep):
                transitions = self.replay_buffer.sample(n)
                S, A = transitions[0], transitions[1] 
                dSS = tf.repeat(S, n, axis=0) - tf.tile(S, [n, 1])
                median_SS = tf.reduce_mean(tf.math.abs(dSS), axis=0)
                if self.sepe_A:
                    median += median_SS + 1e-6
                else:
                    dAA = tf.repeat(A[:,np.newaxis], n, axis=0) - tf.tile(A[:,np.newaxis], [n, 1])
                    median_AA = tf.reduce_mean(tf.math.abs(dAA), axis=0) / 10
                    median += tf.concat([median_SS, median_AA], axis = 0) + 1e-6
#                     X = tf.concat([S, A[:,np.newaxis]], axis=-1)
#                     dxx = tf.repeat(X, n, axis=0) - tf.tile(X, [n, 1])
#                 median += np.mean(tf.math.abs(dxx), axis=0) + 1e-2
#                 median += np.median(tf.reduce_sum(tf.math.abs(dxx), axis = -1)) + 1e-2

            median = median / rep #medians_11_22 / rep, medians_12_33 / rep, medians_12_34 / rep
            return median * 16
    

    def _cal_dist(self, X1 = None, X2 = None, X1_X2 = None, median = None):
        """
        Laplacian Kernel
        pairwise difference: Delta. exp(-diff)
        """
        # shared median
#         if X1_X2 is not None:
#             return tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2), axis=-1) / median) 
#         else:
#             return tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=-1) / median)
        if not self.sepe_A:
            if X1_X2 is not None:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2) / median, axis=-1)) 
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2) / median, axis=-1))
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
        else:
            if X1_X2 is not None:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]) / median, axis=-1))  * tf.cast(X1_X2[:, self.S_dims] == 0, tf.int32)
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]) / median, axis=-1)) * tf.cast(X1[:, self.S_dims] == X2[:, self.S_dims], tf.float64)
                dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :self.S_dims] - X2[:, :self.S_dims]), axis=-1) != 0, tf.float64)
                return dist
  
    def repeat(self, X, rep):
        return tf.repeat(X, rep, axis=0)
    def tile(self, X, rep):
        return tf.tile(X, [rep, 1])
    
    def _compute_loss(self, S, A, SS):
        """n2 = n^2
        
        r left, t right...
        
        _r = repeat = 11
        _t = tile = 1
        order? 
        repeat = 11; tile = 22
        X = [S, A]
        XX = [X, X]
        BS = batch_size
        """
        with tf.device('/gpu:' + str(self.gpu_number)):
            BS = self.BS
            ###########################################################################
            X = tf.concat([S, A[:,np.newaxis]], axis=-1)
            X_r = self.repeat(X, BS)  # n2 x ...
            X_t = self.tile(X, BS) # n2 x ...
            S_r = self.repeat(S, BS) # n2 x ...
            S_t = self.tile(S, BS) # n2 x ...
            SS_r = self.repeat(SS, BS) # n2 x ...
            SS_t = self.tile(SS, BS) # n2 x ...        
            
            S_r_a_list = tf.stack([tf.concat([S_r, tf.expand_dims(tf.repeat(a, BS ** 2), 1)]
                                              , axis = -1) for a in self.A_range], 0) # (n_A, n2)
            S_t_a_list = tf.stack([tf.concat([S_t, tf.expand_dims(tf.repeat(a, BS ** 2), 1)]
                                              , axis = -1) for a in self.A_range], 0) # (n_A, n2)
            SS_r_a_list = tf.stack([tf.concat([SS_r, tf.expand_dims(tf.repeat(a, BS ** 2), 1)]
                                              , axis = -1) for a in self.A_range], 0) # (n_A, n2)
            SS_t_a_list = tf.stack([tf.concat([SS_t, tf.expand_dims(tf.repeat(a, BS ** 2), 1)]
                                              , axis = -1) for a in self.A_range], 0) # (n_A, n2)
            ####################################
            # tf -> numpy -> torch -> numpy -> tf
            SS_r_prob_target = self.target_policy.get_A_prob(SS_r.numpy()) # (n2, n_A) # .cpu()
            SS_r_prob_target = tf.convert_to_tensor(SS_r_prob_target, dtype=tf.float64)

            SS_t_prob_target = self.target_policy.get_A_prob(SS_t.numpy()) # (n2, n_A)
            SS_t_prob_target = tf.convert_to_tensor(SS_t_prob_target, dtype=tf.float64)

            S_t_prob_target = self.target_policy.get_A_prob(S_t.numpy()) # (n2, n_A)
            S_t_prob_target = tf.convert_to_tensor(S_t_prob_target, dtype=tf.float64)

            S_r_prob_target = self.target_policy.get_A_prob(S_r.numpy()) # (n2, n_A)
            S_r_prob_target = tf.convert_to_tensor(S_r_prob_target, dtype=tf.float64)
            ############ omega ############
            omega = self.model.call(X) # n
            """ do we need this? """
            omega = omega / tf.reduce_mean(omega)
            omega = tf.squeeze(omega)
            #################################### part 1 #################################### 
            K_1 = 0
            for i, SS_r_a in enumerate(SS_r_a_list):
                for j, S_t_a in enumerate(S_t_a_list):
                    K_1 += tf.reduce_mean(self._cal_dist(X1 = SS_r_a, X2 = S_t_a
                                                         , median = self.median) * SS_r_prob_target[:, i] * S_t_prob_target[:, j] * self.repeat(omega, BS))
            K_1 *= (2 * self.gamma * (1 - self.gamma))
            
            dxx_2 = tf.stack([self._cal_dist(X1 = X_r, X2 = S_t_a, median = self.median) for S_t_a in S_t_a_list], 1)
            K_2 = tf.reduce_sum(dxx_2 * S_t_prob_target, 1) * self.repeat(omega, BS)
            K_2 = tf.reduce_mean(K_2) 
            K_2 *= (2 * (1 - self.gamma))

            part1 = K_1 - K_2
            #################################### part 2 #################################### 
            omega_r_t = self.repeat(omega, BS) * tf.squeeze(self.tile(omega[:, tf.newaxis], BS)) # n2
            ########
            part2_1 = 0
            for i, SS_r_a in enumerate(SS_r_a_list):
                for j, SS_t_a in enumerate(SS_t_a_list):
                    part2_1 += tf.reduce_mean(self._cal_dist(X1 = SS_r_a
                                                             , X2 = SS_t_a, median = self.median) * SS_r_prob_target[:, i] * SS_t_prob_target[:, j] * omega_r_t)
            part2_1 *= self.gamma ** 2
            ########
            part2_3 = self._cal_dist(X1 = X_r, X2 = X_t, median = self.median)  # n4 # rrr_rrt - ttr_ttt
            part2_3 = tf.reduce_mean(omega_r_t * part2_3)
            ########
            """ not stochastic? can save some computational resources? """ 
            part2_2 = 0
            for i, SS_r_a in enumerate(SS_r_a_list):
                part2_2 += tf.reduce_mean(self._cal_dist(X1 = SS_r_a, X2 = X_t, median = self.median) * SS_r_prob_target[:, i] * omega_r_t)

            part2_2 *= (2 * self.gamma)

            part2 = part2_1 - part2_2 + part2_3
            #################################### part 3 ####################################
            part3 = 0
            for i, S_r_a in enumerate(S_r_a_list):
                for j, S_t_a in enumerate(S_t_a_list):
                    part3 += tf.reduce_mean(self._cal_dist(X1 = S_r_a, X2 = S_t_a, median = self.median) * S_r_prob_target[:, i] * S_t_prob_target[:, j])
            part3 *= (1 - self.gamma) ** 2
            ##################################### final loss ######################################################
            loss = part1 + part2 + part3
#             print(part1.numpy() * 1e3, part2.numpy() * 1e3, part3.numpy() * 1e3, K_1, K_2) # , part2_1.numpy() * 1e3, part2_2.numpy() * 1e3, part2_3.numpy() * 1e3
            return loss * 10000
    ############################################################################################################################################################################################################################################################################################################################################################################################################
    def fit(self, batch_size=32, gamma=0.99, max_iter=100, print_freq = 20, tolerance = 5):
        self.gamma = gamma
        self.median = self._compute_medians()
        # , self.medians_n3, self.medians_n4
        self.BS = batch_size
        cnt_tolerance = 0
        opt_loss = 1e10
        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            S, A, SS = transitions[0], transitions[1], transitions[3]
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                loss = self._compute_loss(S, A, SS)
            with tf.device('/gpu:' + str(self.gpu_number)):
                dw = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
                self.losses.append(loss.numpy())
            if i % 5 == 0 and i >= 10:
                mean_loss = np.mean(self.losses[(i - 10):i])
                if mean_loss / opt_loss - 1 > -0.01:
                    cnt_tolerance += 1
                if mean_loss < opt_loss:
                    opt_loss = mean_loss
                    cnt_tolerance = 0
                if mean_loss < 0 or mean_loss < self.losses[0] / 10:
                    break
            if i % print_freq == 0 and i >= 10:
                print("omega_SA training {}/{} DONE! loss = {:.5f}".format(i, max_iter, mean_loss))
            if cnt_tolerance >= tolerance:
                break 

        self.model(np.random.randn(2, self.model.input_dim))
        
    def predict(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        if inputs.shape[0] > batch_size:
            n_batch = inputs.shape[0] // batch_size + 1
            input_batches = np.array_split(inputs, n_batch)
            return np.vstack([self.model.call(inputs).cpu().numpy() for inputs in input_batches])
        else:
            return self.model.call(inputs).cpu().numpy() 
######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class Omega_SA_Model(tf.keras.Model):
    ''' weights = self.model(S_r, A_r, S_t)
    initial? a random different?
    
    Input for `VisitationRatioModel()`; NN parameterization
    '''
    def __init__(self, S_dims, h_dims, A_dims = 1, gpu_number = 0):
        super(Omega_SA_Model, self).__init__()
        self.gpu_number = gpu_number
        self.hidden_dims = h_dims
        self.seed = 42
        self.input_dim = S_dims + A_dims
        self.reset()
        
    def reset(self):
        
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]

        self.w1 = self.xavier_var_creator(self.input_shape1, name = "w1")
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64), name = "b1")

        self.w2 = self.xavier_var_creator(self.input_shape2, name = "w2")
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b2")

        self.w21 = self.xavier_var_creator(self.input_shape2, name = "w21")
        self.b21 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b21")

        self.w22 = self.xavier_var_creator(self.input_shape2, name = "w22")
        self.b22 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64), name = "b22")

        self.w3 = self.xavier_var_creator(self.input_shape3, name = "w3")
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64), name = "b3")

    def xavier_var_creator(self, input_shape, name = "w3", is_vector = False):
        tf.random.set_seed(self.seed)
        if is_vector:
            xavier_stddev = 1.0 / tf.sqrt(input_shape / 2.0) / 10
        else:
            xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0) / 10
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var

    def call(self, inputs):
        """
        inputs are concatenations of S, A, S_t, A_t = [r,r,t]
        where do we build the model?
        parameter for the depth!!!
        """
        z = tf.cast(inputs, tf.float64)
        h1 = tf.nn.leaky_relu(tf.matmul(z, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        h2 = tf.nn.relu(tf.matmul(h2, self.w21) + self.b21)
        h2 = tf.nn.relu(tf.matmul(h2, self.w22) + self.b22)

        out = (tf.matmul(h2, self.w3) + self.b3) #/ 100 # o.w., the initialization will die
#         out = tf.matmul(h1, self.w3) + self.b3
#         out = tf.nn.leaky_relu(out)
        out = tf.math.log(1.00001 + tf.exp(out))
        out = tf.clip_by_value(out, 0, 10)
        return out

    def predict_4_VE(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                return np.vstack([self.call(inputs).cpu().numpy() for inputs in input_batches])
            else:
                return self.call(inputs).cpu().numpy() 

################################################################################################################################################################
    #     def _normalize(self, weights, BS):
#         """
#         check !!!
#         """
#         weights = tf.reshape(weights, [BS, BS])
#         weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
#         weights = weights / weights_sum
#         return tf.reshape(weights, [BS**2])
        
