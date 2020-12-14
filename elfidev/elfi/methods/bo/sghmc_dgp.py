import tensorflow as tf
import numpy as np
import gpflow

from scipy.cluster.vq import kmeans2

#from tensorflow.python import debug as tf_debug


class BaseModel(object):
    def __init__(self, X, Y, Us, minibatch_size, window_size):
        self.X_placeholder = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        self.Y_placeholder = tf.placeholder(tf.float64, shape=[None, Y.shape[1]])
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.Us = Us
        self.prev_vals = None
        self.minibatch_size = min(minibatch_size, self.N)
        self.data_iter = 0
        self.window_size = window_size
        self.all_samples = []
        self.window = []
        self.posterior_samples = []
        self.sample_op = None
        self.burn_in_op = None


    def generate_update_step(self, nll, epsilon, mdecay, sghmc_pars=None):
        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        if sghmc_pars is not None:
            prev_xi = sghmc_pars[0]
            prev_g = sghmc_pars[1]
            prev_g2 = sghmc_pars[2]
            prev_p = sghmc_pars[3]
            
        grads = tf.gradients(nll, self.Us)
        self.temp_vars = []
        layer = 0
        self.hypers = [[] for i in range(4)]
        for theta, grad in zip(self.Us, grads):
            #grad = tf.clip_by_value(grad, -1, 1)
            if sghmc_pars is None:
                xi = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
                g = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
                g2 = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
                p = tf.Variable(tf.zeros_like(theta), dtype=tf.float64, trainable=False)
            else:
                # print(layer)
                xi = tf.Variable(prev_xi[layer], dtype=tf.float64, trainable=False)
                g = tf.Variable(prev_g[layer], dtype=tf.float64, trainable=False)
                g2 = tf.Variable(prev_g2[layer], dtype=tf.float64, trainable=False)
                p = tf.Variable(prev_p[layer], dtype=tf.float64, trainable=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (tf.sqrt(tf.abs(g2) + 1e-16) + 1e-16)

            self.xi, self.g, self.g2, self.r_t, self.Minv = xi, g, g2, r_t, Minv
            self.xi_t, self.g_t, self.g2_t = xi_t, g_t, g2_t

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = epsilon / tf.sqrt(tf.cast(self.N, tf.float64))
            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16))
            sample_t = tf.random_normal(tf.shape(theta), dtype=tf.float64, seed = self.seed) * sigma
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            self.epsilon_scaled, self.noise_scale, self.sigma  = epsilon_scaled, noise_scale, sigma
            self.sample_t, self.theta_t, self.theta, self.p, self.p_t = sample_t, theta_t, theta, p, p_t

            # self.temp_vars.extend([xi, g, g2, p, r_t, g_t, g2_t, xi_t, Minv])
            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

            self.hypers[0].append(self.xi_t)
            self.hypers[1].append(self.g_t)
            self.hypers[2].append(self.g2_t)
            self.hypers[3].append(self.p_t)
            layer += 1

        self.sample_op = [tf.assign(U, U_t) for U, U_t in sample_updates]
        self.burn_in_op = [tf.assign(var, var_t) for var, var_t in burn_in_updates + sample_updates]
        

    def reset(self, pr=None):
        kern = self.kernels
        lik = self.likelihood
        k_lengthscales = [l.lengthscales._read_parameter_tensor(self.session) for l in self.kernels]
        k_variance = [l.variance._read_parameter_tensor(self.session) for l in self.kernels]
        
        l_var = self.likelihood.variance._read_parameter_tensor(self.session)
        
        print()
        print('Lengthscales, variance, noise_variance (sghmc_dgp.py):')
        print(k_lengthscales)
        print(k_variance)
        print(l_var)
        
        Zs = [self.session.run(l.Z) for l in self.layers]
        Us = [self.session.run(l.U) for l in self.layers]

        tf.reset_default_graph()
        self.session.close()

        return kern, lik, Zs, Us, k_lengthscales, k_variance, l_var, self.sghmc_pars


    def get_params(self):
        k_lengthscales = [l.lengthscales._read_parameter_tensor(self.session) for l in self.kernels]
        k_variance = [l.variance._read_parameter_tensor(self.session) for l in self.kernels]
        
        l_var = self.likelihood.variance._read_parameter_tensor(self.session)
        
        Zs = [self.session.run(l.Z) for l in self.layers]
        Us = [self.session.run(l.U) for l in self.layers]
        return Zs, Us, k_lengthscales, k_variance, l_var, self.sghmc_pars


    def get_minibatch(self):
        assert self.N >= self.minibatch_size
        if self.N == self.minibatch_size:
            return self.X, self.Y

        if self.N < self.data_iter + self.minibatch_size:
            shuffle = np.random.permutation(self.N)
            self.X = self.X[shuffle, :]
            self.Y = self.Y[shuffle, :]
            self.data_iter = 0

        X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
        Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
        self.data_iter += self.minibatch_size
        return X_batch, Y_batch


    def collect_samples(self, num, spacing):
        self.posterior_samples = []
        for i in range(num):
            for j in range(spacing):
                X_batch, Y_batch = self.get_minibatch()
                feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
                _, xi, g, g2, p, r_t, g_t, g2_t, xi_t, Minv, epsilon_scaled, noise_scale, sigma, sample_t, p_t, theta_t, theta, self.sghmc_pars = \
                   self.session.run((self.sample_op, self.xi, self.g, self.g2, self.p, self.r_t, \
                                     self.g_t, self.g2_t, self.xi_t, self.Minv, self.epsilon_scaled, \
                                     self.noise_scale, self.sigma, self.sample_t, self.p_t, self.theta_t, self.theta, self.hypers), feed_dict=feed_dict)
            
            values = self.session.run((self.Us))
            
            '''if np.isnan(values).any():
                print("Collect samples!")
                print('xi')
                print(self.sghmc_pars[0])
                print('g')
                print(self.sghmc_pars[1])
                print('g2')
                print(self.sghmc_pars[2])
                print('p')
                print(self.sghmc_pars[3])
                # print([xi, g, g2, r_t, g_t, g2_t, xi_t, Minv, epsilon_scaled, noise_scale, sigma, sample_t, p, p_t, theta, theta_t])
                # print(self.session.run(self.temp_vars), feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch})
                #raise ValueError

            self.prev_values = values'''

            sample = {}
            for U, value in zip(self.Us, values):
                sample[U] = value

            temp = list()
            for U in self.Us:
                temp.append(sample[U])
            self.all_samples.append(temp)
            self.posterior_samples.append(sample)


    def sghmc_step(self):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        _, self.sghmc_pars = self.session.run((self.burn_in_op, self.hypers), feed_dict=feed_dict)

        values = self.session.run((self.Us))
        '''if np.isnan(values).any():
            print("SGHMC_STEP!")
            print('xi')
            print(self.sghmc_pars[0])
            print('g')
            print(self.sghmc_pars[1])
            print('g2')
            print(self.sghmc_pars[2])
            print('p')
            print(self.sghmc_pars[3])
            # print(values)
            # print('Previous')
            # print(self.prev_values)
            #bool_array = np.isnan(values)
            # print(bool_array)
            #for l in range(0, len(bool_array)):
            #    values[l][bool_array[l]] = self.prev_values[l][bool_array[l]]
            # print('Changed')
            # print(values)

        self.prev_values = values'''
        sample = {}
        for U, value in zip(self.Us, values):
            sample[U] = value

        temp = list()
        for U in self.Us:
            temp.append(sample[U])
        self.all_samples.append(temp)

        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]


    def train_hypers(self):
        return
        '''X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        i = np.random.randint(len(self.window))
        feed_dict.update(self.window[i])
        self.session.run(self.hyper_train_op, feed_dict=feed_dict)'''


    def get_sample_performance(self, posterior=False):
        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        if posterior:
            feed_dict.update(np.random.choice(self.posterior_samples))
        mll, prior = self.session.run((self.log_likelihood, self.prior), feed_dict=feed_dict)
        mll = np.mean(mll, 0)
        prior = prior / self.X.shape[0]
        return mll, prior



class Layer(object):
    def __init__(self, bounds, kern, outputs, n_inducing, fixed_mean, X, Z, U):
        self.inputs, self.outputs, self.kernel = kern.input_dim, outputs, kern
        self.M, self.fixed_mean = n_inducing, fixed_mean

        # initialize Zs with data:
        '''if Z is None:
            # print(kmeans2(X, self.M, minit='points')[0]) minit = ‘random’ Gaussian
            Z = kmeans2(X, self.M, minit='points')[0]
            self.Z = tf.Variable(Z, trainable=False, dtype=tf.float64, name='Z')
        else:
            M_new = self.M - Z.shape[0]
            old_Z = Z
            new_Z = X[-M_new:, :]
            Z = np.vstack((old_Z, new_Z))
            self.Z = tf.Variable(Z, dtype=tf.float64, trainable=False, name='Z')

        if U is None:
            self.U = tf.Variable(np.zeros((self.M, self.outputs)), dtype=tf.float64, trainable=False, name='U')
        else:
            old_U = U
            new_U = np.zeros((M_new, self.outputs))
            U = np.vstack((old_U, new_U))
            self.U = tf.Variable(U, dtype=tf.float64, trainable=False, name='U')'''

        # initialize Zs with a grid:
        if Z is None:
            # print(kmeans2(X, self.M, minit='points')[0]) minit = ‘random’ Gaussian
            min_x, max_x = bounds[0]
            x_mean = (max_x + min_x) / 2.0
            x_std = np.abs(max_x - x_mean)
            min_x = (min_x - x_mean) / x_std
            max_x = (max_x - x_mean) / x_std
        
            # Z = kmeans2(X, self.M, minit='random')[0]
            # print(Z)
            Z = np.linspace(min_x, max_x, num = self.M * X.shape[1])
            Z = Z.reshape((-1, X.shape[1]))
            # print(Z)
            self.Z = tf.Variable(Z, trainable=False, dtype=tf.float64, name='Z')
        else:
            self.Z = tf.Variable(Z, dtype=tf.float64, trainable=False, name='Z')

        if U is None:
            self.U = tf.Variable(np.ones((self.M, self.outputs)) * (-1.0), dtype=tf.float64, trainable=False, name='U')
        else:
            self.U = tf.Variable(U, dtype=tf.float64, trainable=False, name='U')

        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        #print(self.Z.shape)
        #print(self.U.shape)
        #print(X.shape)
        #print(self.outputs)

        
    def conditional(self, X):
        # Caching the covariance matrix from the sghmc steps gives a significant speedup. This is not being done here.
        custom_config = gpflow.settings.get_settings()
        custom_config.numerics.jitter_level = 1e-8

        with gpflow.settings.temp_settings(custom_config):
            # print(gpflow.settings.jitter)
            mean, var = gpflow.conditionals.conditional(X, self.Z, self.kernel, self.U, full_cov=True, white=True)
            mean_uw, var_uw = gpflow.conditionals.conditional(X, self.Z, self.kernel, self.U, full_cov=True, white=False)

        if self.fixed_mean:
            mean += tf.matmul(X, tf.cast(self.mean, tf.float64))

        return mean, var, mean_uw, var_uw
    

    def prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0



class DGP(BaseModel):
    def propagate(self, X):
        Fs, Fmeans, Fvars = [X, ], [], []
        Fs_uw, Fmeans_uw, Fvars_uw = [], [], []

        for layer in self.layers:
            mean, var, mean_uw, var_uw = layer.conditional(Fs[-1]) # NxR, RxNxN
            N = tf.shape(mean)[0] # number of points in a layer

            # std that influences F
            eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float64) # NxR
            eps = tf.matrix_transpose(eps) # RxN
            eps = tf.expand_dims(eps, -1) # RxNx1
            eps = tf.tile(eps, [1, 1, N]) # RxNxN
            
            # std = tf.sqrt(var) # RxNxN
            # std = tf.square(var)
            eps_std = tf.math.multiply(eps, var) # RxNxN
            eps_std = tf.math.reduce_sum(eps_std, axis = 1) # RxN
            eps_std = tf.matrix_transpose(eps_std) # NxR

            # var that we will return
            var = tf.matrix_diag_part(var) # RxN
            var = tf.matrix_transpose(var) # NxR
            # var = tf.square(std)
            
            F = mean + eps_std
            
            eps_std_uw = tf.math.multiply(eps, var_uw) # RxNxN
            eps_std_uw = tf.math.reduce_sum(eps_std_uw, axis = 1) # RxN
            eps_std_uw = tf.matrix_transpose(eps_std_uw) # NxR

            var_uw = tf.matrix_diag_part(var_uw) # RxN
            var_uw = tf.matrix_transpose(var_uw) # NxR

            F_uw = mean_uw + eps_std_uw # triangular solve? see -->
            
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

            Fs_uw.append(F_uw)
            Fmeans_uw.append(mean_uw)
            Fvars_uw.append(var_uw)

        return Fs, Fmeans, Fvars, Fs_uw, Fmeans_uw, Fvars_uw

    
    def __init__(self, X, Y, bounds, n_inducing, kernels, likelihood, Zs, Us, minibatch_size, window_size,
                 adam_lr=0.01, epsilon=0.01, mdecay=0.05, seed=None, sghmc_pars=None):
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size
        self.prev_values = None

        self.seed = seed
        if seed is not None:
            self.seed = seed
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        n_layers = len(kernels)
        N = X.shape[0]

        self.layers = []
        self.kernels_ins = {}
        X_running = X.copy()

        if Zs is None or Us is None:
            Zs = [None] * n_layers
            Us = [None] * n_layers

        for l in range(n_layers):
            outputs = self.kernels[l+1].input_dim if l+1 < n_layers else Y.shape[1]
            self.layers.append(Layer(bounds, self.kernels[l], outputs, n_inducing, fixed_mean=(l+1 < n_layers), X=X_running, Z=Zs[l], U=Us[l]))
            X_running = np.matmul(X_running, self.layers[-1].mean)
            self.kernels[l].compile(session = self.session)
            self.kernels_ins = {**self.kernels_ins, **self.kernels[l].initializable_feeds}

        self.likelihood.compile(session = self.session)
        self.kernels_ins = {**self.kernels_ins, **self.likelihood.initializable_feeds}

        super().__init__(X, Y, [l.U for l in self.layers], minibatch_size, window_size)
        
        self.f, self.fmeans, self.fvars, self.f_uw, self.fmeans_uw, self.fvars_uw = self.propagate(self.X_placeholder)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])

        self.prior = tf.add_n([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)

        # compute gradients of a mean and a var at a point X
        self.grad_means = tf.gradients(self.y_mean, self.X_placeholder)
        self.grad_vars = tf.gradients(self.y_var, self.X_placeholder)

        self.nll = - tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
                   - (self.prior / N)

        self.generate_update_step(self.nll, epsilon, mdecay, sghmc_pars)

        self.adam = tf.train.AdamOptimizer(adam_lr)
        self.hyper_train_op = self.adam.minimize(self.nll)
        
        init_op = tf.global_variables_initializer()
        self.session.run(init_op, feed_dict=self.kernels_ins)


    def predict_y(self, X, S):
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return np.stack(ms, 0), np.stack(vs, 0)


    def get_gradients(self, X):
        # this returns a list of gradients [(x.shape[0], x.shape[1])], but the acquisition function needs
        # only one element (the only element in the list)
        feed_dict = {self.X_placeholder: X}
        grad_mean, grad_var = self.session.run((self.grad_means, self.grad_vars), feed_dict=feed_dict)
        return grad_mean[0], grad_var[0]


    def get_posterior(self, X, i):
        # get prediction for exactly one poster sample
        feed_dict = {self.X_placeholder: X}
        feed_dict.update(self.posterior_samples[i])
        f = self.session.run((self.f[-1]), feed_dict=feed_dict)
        # m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
        return f


    def get_posterior_fs_for_layer(self, X, s, l):
        # given an input, gives the output for the ith layer
        feed_dict = {self.X_placeholder: X}
        feed_dict = {self.f[l]: X}
        feed_dict.update(self.posterior_samples[s])
        f_uw, m_uw, v_uw, Z, U = self.session.run((self.f_uw[l], self.fmeans_uw[l], self.fvars_uw[l], self.layers[l].Z, self.layers[l].U), feed_dict=feed_dict)
        return f_uw, m_uw, v_uw, Z, U


    def get_points_posterior(self):
        S = len(self.posterior_samples)
        Zs = self.X
        # print(self.X)
        Us = []
        result = [Zs]
        for l in range(len(self.layers)):
            for i in range(S):
                feed_dict = {self.f[l]: Zs}
                feed_dict.update(self.posterior_samples[i])
                m, v = self.session.run((self.fmeans[l], self.fvars[l]), feed_dict=feed_dict)
                eps = tf.random_normal(tf.shape(m), dtype=tf.float64)
                U = m + eps * tf.sqrt(v)
                Us.append(U)
            Zs = tf.concat(Us, 0).eval(session=self.session)
            Us = []
            result.append(Zs)
        return result[::-1]

