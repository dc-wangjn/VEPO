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

class VisitationRatioModel_init_SSA():
    """ w(S', A'; S) <- model
    just change (SASA) to (SAS)
    see the derivations on page 2, the supplement of VE
    should be almost the same with SASA
    
    can we just delete? IS? 
    the same estimation?
    """
    def __init__(self, replay_buffer
                 , target_policy, A_range
                 , h_dims, A_dims = 1, gpu_number = 0
                 , lr = 1e-3, w_clipping_val = 0.5, w_clipping_norm = 1.0, beta_1=0.5):
        self.model = Omega_SSA_Model(replay_buffer.S_dims, h_dims, A_dims, gpu_number)
        self.S_dims = replay_buffer.S_dims
        self.A_dims = A_dims
        self.gpu_number = gpu_number
        self.replay_buffer = replay_buffer
        self.target_policy = target_policy
        self.A_range = A_range
        self.n_A = len(A_range)
        self.losses = []    
        self.sepe_A = 0
        self.single_median = False
        self.optimizer = tf.keras.optimizers.Adam(lr = lr, beta_1 = beta_1, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
        
    def _compute_medians(self, n= 32, rep = 20):
        # to save memory, we do it iteratively
        with tf.device('/gpu:' + str(self.gpu_number)):
            if self.single_median:
                medians_11_22, medians_12_33, medians_12_34 = 0, 0, 0
            else:
                medians_11_22, medians_12_33, medians_12_34 = [tf.zeros([(self.S_dims + self.A_dims + self.S_dims)], tf.float64) for _ in range(3)]
            for i in range(rep):
                transitions = self.replay_buffer.sample(n)
                S, A = transitions[0], transitions[1] 
                X = tf.concat([S, A[:,np.newaxis]], axis=-1)
                #################
                XS = tf.concat([X, S], axis=-1) # n
                dxx = tf.repeat(XS, n, axis=0) - tf.tile(XS, [n, 1])
                if self.single_median:
                    medians_11_22 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_11_22 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5
                #################
                Xr_St = tf.concat([tf.repeat(X, n, axis=0),tf.tile(S, [n, 1])], axis=-1) # n2
                dxx = tf.repeat(Xr_St, n, axis=0) - tf.tile(XS, [n ** 2, 1]) # n3 x ...
                if self.single_median:
                    medians_12_33 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_12_33 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p
                #################
                dxx = tf.repeat(Xr_St, n ** 2, axis=0) - tf.tile(Xr_St, [n ** 2, 1]) # n4 x ...
                if self.single_median:
                    medians_12_34 += np.mean(tf.reduce_sum(tf.math.abs(dxx), axis = -1), axis=0) + 1e-5
                else:
                    medians_12_34 += np.mean(tf.math.abs(dxx), axis=0) + 1e-5 # p
        dim = self.S_dims * 2 + self.A_dims
        return medians_11_22 / rep * dim, medians_12_33 / rep * dim, medians_12_34 / rep * dim
    
    """ median, laplacian, etc, are all not correct """
    
    
    def _cal_dist(self, X1 = None, X2 = None, X1_X2 = None, median = None):
        """
        Laplacian Kernel
        pairwise difference: Delta. exp(-diff)
        """
        
        if X1_X2 is not None:
            if self.single_median:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2), axis=-1) / median ) 
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1_X2 / median), axis=-1)) 
            dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1_X2[:, :(self.S_dims * 2)]), axis=-1) != 0, tf.float64)
            return dist
        else:
            if self.single_median:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=-1) / median) 
            else:
                dist = tf.exp(-tf.math.reduce_sum(tf.math.abs((X1 - X2) / median), axis=-1))
            dist = dist * tf.cast(tf.math.reduce_sum(tf.math.abs(X1[:, :(self.S_dims * 2)] - X2[:, :(self.S_dims * 2)]), axis=-1) != 0, tf.float64)
            return dist
    
    def repeat(self, X, rep):
        return tf.repeat(X, rep, axis=0)
    def tile(self, X, rep):
        return tf.tile(X, [rep, 1])
    def _compute_loss(self, S, A, SS, gamma):
        """n2 = n^2
        
        _r = repeat = 11
        _t = tile = 1
        order? 
        repeat = 11; tile = 22
        X = [S, A]
        XX = [X, X]
        BS = batch_size
        
        S, A | S_0 = S
        S = 1 = t
        S_t is the initial state
        """
        with tf.device('/gpu:' + str(self.gpu_number)):
            BS = self.BS
            n = self.BS
            ###########################################################################
            # X_r = tf.concat([self.repeat(S, BS), tf.expand_dims(tf.repeat(A, BS), 1)], axis=-1)  # n2 x ...
            # X_t = tf.concat([self.tile(S, BS), self.tile(tf.expand_dims(A, 1), BS)], axis=-1)  # n2 x ...
            # S_r = self.repeat(S, BS)
            # XX = tf.concat([S, A[:,None], S, A[:,None]], axis=-1) # n1 x ...
            S_t = self.tile(S, BS)
            S_r = self.repeat(S, BS) # n2 x ...       
            X = tf.concat([S, A[:,np.newaxis]], axis=-1)
            XS = tf.concat([X, S], axis=-1) # n

            Xr_St = tf.concat([tf.repeat(X, n, axis=0),tf.tile(S, [n, 1])], axis=-1) # n2
            S_r_a_list = tf.stack([tf.concat([S_r, tf.expand_dims(tf.repeat(a, BS ** 2), 1)]
                                              , axis = -1) for a in self.A_range], 0) # (n_A, n2)
            ###### probability ######
            S_r_prob_target = tf.convert_to_tensor(self.target_policy.get_A_prob(S_r), dtype=tf.float64) # (n2, n_A)

            ###### omega (normalization) ######

            omega_r_t = self.model.call(Xr_St) # n2
            # omega_r_t = tf.transpose(omega_r_t)
            norm_omega = tf.reduce_mean(tf.reshape(omega_r_t, [n, n]), 0)
            omega_r_t /= (tf.tile(tf.expand_dims(norm_omega, 1), [n,1]))
            omega_r_t = tf.squeeze(omega_r_t)

#             omega_r_t = self.model.call(Xr_St) # n2
#             norm_omega = tf.reduce_mean(tf.reshape(omega_r_t, [n, n]), 1)
#             omega_r_t = tf.squeeze(omega_r_t) / (tf.repeat(tf.expand_dims(norm_omega, 1), n))

            # n = batch_size
            # self.del_dup = tf.cast(tf.repeat(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1)), n), tf.float64)

            ##########################################################################################
            #################################### part 1, n3 #################################### 
            dxx_ra_t_2_2 = tf.stack([self.repeat(tf.concat([S_r_a, S_t], axis = -1), BS) - \
                                     self.tile(XS, BS ** 2) for S_r_a in S_r_a_list], 0) # (n_A, n3), rr_rt - tt_tt
            K_ra_t_2_2 = tf.stack([self._cal_dist(X1_X2 = a, median = self.medians_n3) for a in dxx_ra_t_2_2], 0) # (n_A, n3)
            E_K_ra_t_2_2 = tf.reduce_sum(self.repeat(S_r_prob_target, BS) * tf.transpose(K_ra_t_2_2), 1) # (n3, n_A)
            K_r_t_2_2 = self._cal_dist(X1 = self.repeat(Xr_St, BS), X2 = self.tile(XS, BS ** 2), median = self.medians_n3)
            
            part1_bef_sum = tf.squeeze(self.repeat(omega_r_t, BS)) * (gamma * E_K_ra_t_2_2 - K_r_t_2_2) * self.del_dup_n3
            part1 = 2 * (1 - gamma) * tf.reduce_mean(part1_bef_sum) 
            #################################### part 2, n4 #################################### 
            omega_r_t_22_2 = tf.squeeze(self.repeat(omega_r_t, BS ** 2)) * tf.squeeze(self.tile(tf.expand_dims(omega_r_t, 1), BS ** 2)) * self.del_dup_n4 # n4
            ######
            part2_1 = 0
            for i, S_r_a1 in enumerate(S_r_a_list):
                for j, S_r_a2 in enumerate(S_r_a_list):
                    K_term = self._cal_dist(X1 = self.tile(tf.concat([S_r_a2, S_t], axis = -1), BS ** 2)
                                          , X2 = tf.repeat(tf.concat([S_r_a1, S_t], axis = -1), BS ** 2, axis=0)
                                          , median = self.medians_n4)
                    K_prob = tf.squeeze(tf.tile(tf.expand_dims(S_r_prob_target[:, j], 1), [BS ** 2, 1])) * \
                                    tf.repeat(S_r_prob_target[:, i], BS ** 2) * K_term
                    part2_1 += gamma ** 2 * tf.reduce_mean(K_prob * omega_r_t_22_2) 
                
            ######
            diff_part2_2 = tf.stack([self.repeat(tf.concat([S_r_a, S_t], axis = -1), BS ** 2) - self.tile(Xr_St, BS**2) for S_r_a in S_r_a_list], axis = 0) # (n_A, n4) # rrr_rrt - ttr_ttt
            K_part2_2 = tf.stack([self._cal_dist(X1_X2 = a, median = self.medians_n4) for a in diff_part2_2], 0) # (n_A, n4)
            E_K_part2_2 = tf.reduce_sum(tf.transpose(K_part2_2) * self.repeat(S_r_prob_target, BS ** 2)
                                           , -1)# (n4, n_A)
            part2_2 = tf.reduce_mean(E_K_part2_2 * omega_r_t_22_2) * 2 * gamma 
            ######
            part2_3 = self._cal_dist(X1 = self.repeat(Xr_St, BS ** 2), X2 = self.tile(Xr_St, BS ** 2), median = self.medians_n4)  # n4 # rrr_rrt - ttr_ttt
            part2_3 = tf.reduce_mean(omega_r_t_22_2 * part2_3)

            part2 = (part2_1 + part2_3 - part2_2)# / BS ** 2
            #################################### part 3 ####################################
            dxx3 = self._cal_dist(X1 = self.repeat(XS, BS)
                                  , X2 = self.tile(XS, BS)
                                  , median = self.medians_n2) # n2
            part3 = (1 - gamma) ** 2 * tf.reduce_mean(dxx3) 
            ##################################### final loss ######################################################
            loss = (part1 + part2 + part3) * 1e3
            return loss
#             return tf.abs(loss)
    ########################################################################################################
    def fit(self, batch_size=32, gamma=0.99, max_iter=100, print_freq = 5, tolerance = 5, rep_once = 3):
        self.medians_n2, self.medians_n3, self.medians_n4 = self._compute_medians()
        self.BS = batch_size
        cnt_tolerance = 0
        opt_loss = 1e10
        n = batch_size
        self.del_dup_n3 = tf.cast(tf.repeat(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1)), n), tf.float64)
        self.del_dup_n4 = tf.cast(tf.repeat(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1)), n ** 2), tf.float64) * tf.cast(tf.tile(tf.reshape(tf.linalg.set_diag(tf.ones((n, n)), tf.zeros(n)), (-1)), [n ** 2]), tf.float64)

        for i in range(max_iter):
            ##### compute loss function #####
            with tf.GradientTape() as tape:
                loss = 0
                for j in range(rep_once):
                    transitions = self.replay_buffer.sample(batch_size)
                    S, A, SS = transitions[0], transitions[1], transitions[3]
                    loss += self._compute_loss(S, A, SS, gamma)
            with tf.device('/gpu:' + str(self.gpu_number)):
                dw = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
                self.losses.append(loss.numpy())
            if i % print_freq == 0 and i >= 10:
                mean_loss = np.mean(self.losses[(i - 10):i])
                print("omega_SSA training {}/{} DONE! loss = {:.5f}".format(i, max_iter, mean_loss))
                if mean_loss / opt_loss - 1 > -0.01:
                    cnt_tolerance += 1
                if mean_loss < opt_loss:
                    opt_loss = mean_loss
                    cnt_tolerance = 0
                if mean_loss < 0 or mean_loss < self.losses[0] / 10:
                    break
            elif i < 10 and i % 3 == 0:
                print("loss for iter {} = {:.2f}".format(i, loss.numpy()))
            if cnt_tolerance >= tolerance:
                break 

        self.model(np.random.randn(2, self.model.input_dim))
    ########################################################################################################
    def predict(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                return np.vstack([self.model.call(inputs).cpu().numpy() for inputs in input_batches])
            else:
                return self.model.call(inputs).cpu().numpy() 
##############################################################################################################################

class Omega_SSA_Model(tf.keras.Model):
    ''' weights = self.model(S_r, A_r, S_t)
    initial? a random different?
    
    Input for `VisitationRatioModel()`; NN parameterization
    '''
    def __init__(self, S_dims, h_dims, A_dims = 1, gpu_number = 0):
        super(Omega_SSA_Model, self).__init__()
        self.hidden_dims = h_dims
        self.gpu_number = gpu_number

        self.input_dim = S_dims + A_dims + S_dims
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



        
    def xavier_var_creator(self, input_shape, name = "w3"):
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0) / 3
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True, name = name)
        return var

    def call(self, inputs):
        """
        inputs: concatenations of S, A, S_t, A_t = [r,r,t]
        outputs: weights?
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
#         out = tf.clip_by_value(out, 0, 10)
        return out


    def predict_4_VE(self, inputs, to_numpy = True, small_batch = True, batch_size = int(1024 * 8 * 16)):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if inputs.shape[0] > batch_size:
                n_batch = inputs.shape[0] // batch_size + 1
                input_batches = np.array_split(inputs, n_batch)
                return np.vstack([self.call(inputs).cpu().numpy() for inputs in input_batches])
            else:
                return self.call(inputs).cpu().numpy() 

    
    #     def _normalize(self, weights, BS):
#         """
#         check !!!
#         """
#         weights = tf.reshape(weights, [BS, BS])
#         weights_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
#         weights = weights / weights_sum
#         return tf.reshape(weights, [BS**2])
        
