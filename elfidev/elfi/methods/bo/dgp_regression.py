"""This module contains an interface for using the DGP with IWVI in ELFI."""
import numpy as np
import tensorflow as tf
import copy
import logging
import pickle

#from elfi.methods.bo.sghmc_dgp import DGP

#import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans2

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.features import InducingPoints
from gpflow.training import NatGradOptimizer, AdamOptimizer
from gpflow.mean_functions import Linear
from gpflow import defer_build

from gpflow.multioutput.features import MixedKernelSharedMof
# from gpflow.multioutput.kernels import SharedMixedMok

import gpflow

from elfi.methods.bo.iwvi.layers import GPLayer, LatentVariableLayer
from elfi.methods.bo.iwvi.temp_workaround import SharedMixedMok
from elfi.methods.bo.iwvi.models import DGP_VI, DGP_IWVI

from elfi.methods.bo.iwvi.sghmc import SGHMC

logger = logging.getLogger(__name__)
logging.getLogger("DGP").setLevel(logging.WARNING)  # DGP logger


class DGPRegression:
    def __init__(self, parameter_names=None, bounds=None, GPlayers=3, LVlayer=True, \
        Ms=50, IW_samples = 5, pred_samples = 100, opt_steps = 20000, q = 0.3):

        '''Initialize DGPRegression.

        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameter nodes. If None, sets dimension to 1.
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `{'parameter_name':(lower, upper), ... }`
            If not supplied, defaults to (0, 1) bounds for all dimensions.
        layers : int, optional
            number of layers in a DGP model.
        Ms: int, optional
            number of inducing points per each layer.
        IW_samples : int, optional
            number of Importance-Weighted samples.
        pred_samples : int, optional
            number of samples are used for predictions and gradients. 
        opt_steps : int, optional
            number of hyperparameter optimization steps
        '''
    
        class ARGS:
            minibatch_size = None
            lr = 5e-3
            lr_decay = 0.99

        class Model(ARGS):
            # 'VI', 'HMC', 'IWAE'
            mode = 'IWAE'
            M = Ms # 100
            likelihood_variance = 0.1
            fix_linear = True
            num_IW_samples = IW_samples # was 20 # 5
            gamma = 5e-2
            gamma_decay = 0.99

            if LVlayer is True:
                configuration = 'L1' # L1
            else:
                configuration = 'G1'
            
            for _ in range(GPlayers): 
                configuration += '_G1'

        self.its = opt_steps
        self._gp = None
        self.session = None
        self.quantile = q

        self.model_type = Model

        self.X = None
        self.Y = None
        self.x_mean = None
        self.x_std = None 
        self.y_mean = None
        self.y_std = None

        if parameter_names is None:
            input_dim = 1
        elif isinstance(parameter_names, (list, tuple)):
            input_dim = len(parameter_names)

        if bounds is None:
            logger.warning('Parameter bounds not specified. Using [0,1] for each parameter.')
            bounds = [(0, 1)] * input_dim
        elif len(bounds) != input_dim:
            raise ValueError(
                'Length of `bounds` ({}) does not match the length of `parameter_names` ({}).'
                .format(len(bounds), input_dim))
        elif isinstance(bounds, dict):
            if len(bounds) == 1:  # might be the case parameter_names=None
                bounds = [bounds[n] for n in bounds.keys()]
            else:
                # turn bounds dict into a list in the same order as parameter_names
                bounds = [bounds[n] for n in parameter_names]
        else:
            raise ValueError("Keyword `bounds` must be a dictionary "
                             "`{'parameter_name': (lower, upper), ... }`")
        self.bounds = bounds
        self.input_dim = input_dim
        self.S = pred_samples 
        self.mlls = []
        self.posterior = None
        return

        
    # -
    def __str__(self):
        """Return GPy's __str__."""
        return self._gp.__str__()

    # -
    def __repr__(self):
        """Return GPy's __str__."""
        return self.__str__()


    # + noiseless is not implemented
    def predict(self, X, noiseless=False):
        """Return the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        noiseless : bool
            whether to include the noise variance or not to the returned variance

        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        X = np.asanyarray(X).reshape((-1, self.input_dim))
        X = (X - self.x_mean) / self.x_std
        # y = self._gp.predict_y_samples(X, self.S)
        # return y * self.y_std + self.y_mean #, v * self.y_std
        # m, v = self._gp.predict_y(X, session=self.session)
        # ms, vs = self._gp.predict_f_multisample(X, self.S)
        # m = np.average(ms, 0)
        # v = np.average(vs + ms**2, 0) - m**2

        fs = self._gp.predict_y_samples(X, self.S)
        # m = np.mean(fs, 0)
        # v = np.var(fs, 0)
        q = np.quantile(fs, self.quantile, axis=0)
        # print(fs.shape)
        # print(q)
        # print(X
        # print(fs)
        q_fs = list()
        for sample in fs:
            q_fs.append(list())
            for i in range(0, len(sample)):
                # print(sample)
                # print(q)
                if sample[i] <= q[i]:
                    el = sample[i]
                else:
                    el = np.array(np.nan)
                # print(el)
                el = el.reshape((1))
                q_fs[-1].append(np.array(el))
                
            q_fs[-1] = np.array(q_fs[-1])

        q_fs = np.array(q_fs)

        # q_fs = np.array(q_fs).reshape(20, 1, 1)
        # q_fs = fs[np.where(fs[i] < q[i] for i in range(0, self.S))]
        
        # print('Q_fs' + str(q_fs.shape))
        # print('Fs' + str(fs.shape))
        # print(q_fs)
        
        # q_fs = [f for f in fs if f < q[0]]
        q_mean = np.nanmean(q_fs, 0)
        q_var = np.nanvar(q_fs, 0)
        
        # print('Fs shape ' + str(fs.shape)) 
        # print('Qmean shape ' + str(q_mean.shape))

        if isinstance(q_mean[0], float) == False:
            q_mean = np.concatenate( q_mean, axis=0 )
            q_var = np.concatenate( q_var, axis=0 )
        # # print(q_mean)
        # print(q_var)
        # print(q_mean.ndim)
        # print(q_mean.shape)
        
        # print(np.concatenate( q_fs, axis=0 ))
        return q_mean * self.y_std + self.y_mean, q_var * self.y_std
        #return m * self.y_std + self.y_mean, v * self.y_std


    def sample_fs(self, X, noiseless=False):
        """Return the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        noiseless : bool
            whether to include the noise variance or not to the returned variance

        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        X = np.asanyarray(X).reshape((-1, self.input_dim))
        X = (X - self.x_mean) / self.x_std
        
        
        if self.model_type.mode == 'SGHMC':   
            fs = self._gp.predict_y_samples(X, self.S)
        else:
            if self.posterior is None:
                self.posterior =  self.sghmc_optimizer.collect_samples(self.session, self.S)
            
            feed_dict = {}
            fs = []
            for s in self.posterior:
                feed_dict.update(s)
                sample_fs = self.session.run(self.Fs, feed_dict=feed_dict)
                fs.append(sample_fs)
            print(self.posterior)
        return fs

        

    # + 
    def predict_mean(self, X):
        """Return the GP model mean function at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(X)[0]


    def predict_var(self, X):
        """Return the GP model variance function at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(X)[1]


    def predictive_gradients(self, X):
        """Return the gradients of the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        tuple
            GP (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)

        """
        X = X.reshape((-1, self.input_dim))
        X = (X - self.x_mean) / self.x_std

        feed_dict = {self.X_placeholder: X}
        mean_grad_val, var_grad_val = self.session.run((self.mean_grad, self.var_grad),
                                                      feed_dict=feed_dict)
        # qmean_grad_val, qvar_grad_val = self.session.run((self.qmean_grad, self.qvar_grad),
        #                                                 feed_dict=feed_dict)
        # return qmean_grad_val[0], qvar_grad_val[0]
        return  mean_grad_val[0], var_grad_val[0]


    # -
    def predictive_gradient_mean(self, X):
        """Return the gradient of the GP model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)
        """
        return self.predictive_gradients(X)


    # + if there is a model --> create, if not --> reset
    def _init_gp(self, X, Y, optimize):
        if self.x_mean is None or self.x_std is None:
            min_x, max_x = map(list,zip(*self.bounds))
            min_x, max_x = np.array(min_x), np.array(max_x)
            self.x_mean = (max_x + min_x) / 2.0
            self.x_std = np.abs(max_x - self.x_mean)
        
        self.y_mean = np.mean(Y, 0)
        self.y_std = np.std(Y, 0)
        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std

        cond = False
        if self._gp is not None:
            self._gp.X = X
            self._gp.Y = Y
            cond = False
            

        if optimize == True:

            if self._gp is not None:
                self._gp.clear()
            self._gp = self.build_model(self.model_type, X, Y, conditioning=cond, apply_name=None)
            self._gp.model_name = 'LV-GP-GP'
            self.session = self._gp.enquire_session()
            self._gp.init_op(self.session)

        '''if cond == False:
            self.optimize()

            self.mlls.append(self._gp.compute_log_likelihood())
            print('\n\n')
            print(self.mlls[-1])
            print("Finished training")
            self._gp.anchor(self.session)'''
        #plot_samples(model, path)
        #plot_density(model, path)
            

    # + doesn't use noise_var, mean_function
    def build_model(self, ARGS, X, Y, conditioning=False, apply_name=True,
                    noise_var=None, mean_function=None):

        if conditioning == False:
            N, D = X.shape

            # first layer inducing points
            if N > ARGS.M:
                Z = kmeans2(X, ARGS.M, minit='points')[0]
            else:
                # This is the old way of initializing Zs
                # M_pad = ARGS.M - N
                # Z = np.concatenate([X.copy(), np.random.randn(M_pad, D)], 0)

                # This is the new way of initializing Zs
                min_x, max_x = self.bounds[0]
                min_x = (min_x - self.x_mean) / self.x_std
                max_x = (max_x - self.x_mean) / self.x_std
            
                Z = np.linspace(min_x, max_x, num = ARGS.M) # * X.shape[1])
                Z = Z.reshape((-1, X.shape[1]))
                #print(min_x)
                #print(max_x)
                #print(Z)
                

            #################################### layers
            P = np.linalg.svd(X, full_matrices=False)[2]
            # PX = P.copy()

            layers = []
            # quad_layers = []

            DX = D
            DY = 1

            D_in = D
            D_out = D
            
            with defer_build():

                # variance initialiaztion
                lik = Gaussian()
                lik.variance = ARGS.likelihood_variance

                if len(ARGS.configuration) > 0:
                    for c, d in ARGS.configuration.split('_'):
                        if c == 'G':
                            num_gps = int(d)
                            A = np.zeros((D_in, D_out))
                            D_min = min(D_in, D_out)
                            A[:D_min, :D_min] = np.eye(D_min)
                            mf = Linear(A=A)
                            mf.b.set_trainable(False)

                            def make_kern():
                                k = RBF(D_in, lengthscales=float(D_in) ** 0.5, variance=1., ARD=True)
                                k.variance.set_trainable(False)
                                return k

                            PP = np.zeros((D_out, num_gps))
                            PP[:, :min(num_gps, DX)] = P[:, :min(num_gps, DX)]
                            ZZ = np.random.randn(ARGS.M, D_in)
                            # print(Z.shape)
                            # print(ZZ.shape)
                            ZZ[:, :min(D_in, DX)] = Z[:, :min(D_in, DX)]

                            kern = SharedMixedMok(make_kern(), W=PP)
                            inducing = MixedKernelSharedMof(InducingPoints(ZZ))

                            l = GPLayer(kern, inducing, num_gps, mean_function=mf)
                            if ARGS.fix_linear is True:
                                kern.W.set_trainable(False)
                                mf.set_trainable(False)

                            layers.append(l)

                            D_in = D_out

                        elif c == 'L':
                            d = int(d)
                            D_in += d
                            layers.append(LatentVariableLayer(d, XY_dim=DX+1))

                # kernel initialization
                kern = RBF(D_in, lengthscales=float(D_in)**0.5, variance=1., ARD=True)
                ZZ = np.random.randn(ARGS.M, D_in)
                ZZ[:, :min(D_in, DX)] = Z[:, :min(D_in, DX)]
                layers.append(GPLayer(kern, InducingPoints(ZZ), DY))
                self.layers = layers
                self.lik = lik

            # global_step = tf.Variable(0, dtype=tf.int32)
            # self.global_step = global_step
        else:
            lik = self._gp.likelihood
            layers = self._gp.layers._list
            # val = self.session.run(self.global_step)
            # global_step = tf.Variable(val, dtype=tf.int32)
            # self.global_step = global_step
            self._gp.clear()

   
        with defer_build():
            
            #################################### model
            name = 'Model' if apply_name else None
            

            if ARGS.mode == 'VI':
                model = DGP_VI(X, Y, layers, lik,
                               minibatch_size=ARGS.minibatch_size,
                               name=name)

            elif ARGS.mode == 'SGHMC':
                for layer in layers:
                    if hasattr(layer, 'q_sqrt'):
                        del layer.q_sqrt
                        layer.q_sqrt = None
                        layer.q_mu.set_trainable(False)

                model = DGP_VI(X, Y, layers, lik,
                               minibatch_size=ARGS.minibatch_size,
                               name=name)


            elif ARGS.mode == 'IWAE':
                model = DGP_IWVI(X, Y, layers, lik,
                                 minibatch_size=ARGS.minibatch_size,
                                 num_samples=ARGS.num_IW_samples,
                                 name=name)

        global_step = tf.Variable(0, dtype=tf.int32)
        op_increment = tf.assign_add(global_step, 1)

        if not ('SGHMC' == ARGS.mode):
            for layer in model.layers[:-1]:
                if isinstance(layer, GPLayer):
                    layer.q_sqrt = layer.q_sqrt.read_value() * 1e-5

            model.compile()

            #################################### optimization

            var_list = [[model.layers[-1].q_mu, model.layers[-1].q_sqrt]]

            model.layers[-1].q_mu.set_trainable(False)
            model.layers[-1].q_sqrt.set_trainable(False)

            gamma = tf.cast(tf.train.exponential_decay(ARGS.gamma, global_step, 1000, ARGS.gamma_decay, staircase=True),
                            dtype=tf.float64)
            lr = tf.cast(tf.train.exponential_decay(ARGS.lr, global_step, 1000, ARGS.lr_decay, staircase=True), dtype=tf.float64)

            op_ng = NatGradOptimizer(gamma=gamma).make_optimize_tensor(model, var_list=var_list)

            op_adam = AdamOptimizer(lr).make_optimize_tensor(model)

            def train(s):
                s.run(op_increment)
                s.run(op_ng)
                s.run(op_adam)

            model.train_op = train
            model.init_op = lambda s: s.run(tf.variables_initializer([global_step]))
            model.global_step = global_step

        else:
            model.compile()

            sghmc_vars = []
            for layer in layers:
                if hasattr(layer, 'q_mu'):
                    sghmc_vars.append(layer.q_mu.unconstrained_tensor)

            hyper_train_op = AdamOptimizer(ARGS.lr).make_optimize_tensor(model)

            self.sghmc_optimizer = SGHMC(model, sghmc_vars, hyper_train_op, 100)

            def train_op(s):
                s.run(op_increment),
                self.sghmc_optimizer.sghmc_step(s),
                self.sghmc_optimizer.train_hypers(s)

            model.train_op = train_op
            model.sghmc_optimizer = self.sghmc_optimizer
            def init_op(s):
                epsilon = 0.01
                mdecay = 0.05
                with tf.variable_scope('sghmc'):
                    self.sghmc_optimizer.generate_update_step(epsilon, mdecay)
                v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sghmc')
                s.run(tf.variables_initializer(v))
                s.run(tf.variables_initializer([global_step]))

            # Added jitter due to input matrix invertability problems
            custom_config = gpflow.settings.get_settings()
            custom_config.numerics.jitter_level = 1e-8

            model.init_op = init_op
            model.global_step = global_step

        # build the computation graph for the gradient
        self.X_placeholder = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        self.Fs, Fmu, Fvar = model._build_predict(self.X_placeholder)
        self.mean_grad = tf.gradients(Fmu, self.X_placeholder)
        self.var_grad = tf.gradients(Fvar, self.X_placeholder)

        # calculated the gradient of the mean for the quantile-filtered distribution
        # print(Fs)
        # q = np.quantile(Fs, self.quantile, axis=0)
        # qFs = [f for f in Fs if f < q]
        # q_mean = np.mean(qFs, axis=0)
        # q_var = np.var(qFs, axis=0)
        # self.qmean_grad = tf.gradients(q_mean, self.X_placeholder)
        # self.qvar_grad = tf.gradients(q_var, self.X_placeholder)
                                  
        return model

    
    # +
    def update(self, X, Y, optimize=False):
        """Update the GP model with new data.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize hyperparameters.

        """  
        # Must cast these as 2d for GPy      
        X = X.reshape((-1, self.input_dim))
        Y = Y.reshape((-1, 1))

        if self.X is None or self.Y is None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.r_[self.X, X]
            self.Y = np.r_[self.Y, Y]

        self._init_gp(self.X, self.Y, optimize)

        # print(self.X)
        # print("Optimize? (dgp_regression.py)" + str(optimize))

        if optimize:
            self.optimize()

        if self._gp is not None:
            self.mlls.append(self._gp.compute_log_likelihood())
            print('\nMLL of the iteration: ' + str(self.mlls[-1]) + '\n')
            self._gp.anchor(self.session)


    def optimize(self):
        """Optimize DGP hyperparameters."""
        logger.debug("Optimizing DGP hyperparameters")
        for it in range(self.its):
            self._gp.train_op(self.session)

            #if it % 100 == 0 :
            #    self.mlls.append(self._gp.compute_log_likelihood())

    def get_HMC_samples(self):
        return self.sghmc_optimizer.collect_samples(self.session, num=1000, spacing = 50)
        

    def get_posterior(self, x, size):
        x = np.asanyarray(x).reshape((-1, self.input_dim))
        x = (x - self.x_mean) / self.x_std
        # f = self._gp.predict_y_samples(x, size, session=self.session)

        fs = self._gp.predict_y_samples(x, size, session=self.session)
        
        '''if self.model_type.mode == None: 
            if self.posterior is None:
                self.posterior =  self.hmc_optimizer.collect_samples(self.session, self.S)

            fs = []
            for s in self.posterior:
                feed_dict = {self.X_placeholder: x}
                feed_dict.update(s)
                sample_fs = self.session.run(self.Fmu, feed_dict=feed_dict)
                fs.append(sample_fs)'''
        return fs * self.y_std + self.y_mean

    
    def plot_mlls(self):
        if self._gp is None:
            return
        
        x = list()
        for i in range(0, len(self.mlls)):
            x.append(i+1)
        plt.xticks(np.arange(min(x), max(x)+1))
        plt.grid(color='grey', linestyle='-', linewidth=0.5)
        plt.plot(x, self.mlls, color='blue', label='LogLik')
        plt.legend(loc='upper left')
        return

    '''# Not implented!
    def save_model(self, file = 'model.pkl'):
        with open(file, 'wb') as f:
            pickle.dump([self.X, self.Y, self.model_type], f)

    # Not implented!
    def load_model(self, file = 'model.pkl'):
        with open(file) as f:
            X, Y, self.model_type = pickle.load(f)
            self._init_gp(self, X, Y, optimize=False)'''

    def del_graph(self):
        gpflow.reset_default_graph_and_session()
        return
        
    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return self._gp.X.size 

    # +
    '''@property
    def X(self):
        """Return input evidence."""
        return self._gp.X.value * self.x_std + self.x_mean

    # +
    @property
    def Y(self):
        """Return output evidence."""
        return self._gp.Y.value * self.y_std + self.y_mean'''

    @property
    def noise(self):
        """Return the noise."""
        return self._gp.Gaussian_noise.variance[0]

    # +
    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    # +
    def copy(self):
        """Return a copy of current instance."""
        kopy = copy.copy(self)
        return kopy

    def __copy__(self):
        """Return a copy of current instance."""
        return self.copy()

