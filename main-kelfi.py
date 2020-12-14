'''KELFI code is adapated from https://github.com/Kelvin-Hsu/kelfi '''

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle, time, os, sys
from os import path

import elfi
from elfi.examples import dgp_funcs, bdm_dgp, navworld

from kelfi.utils import halton_sequence
from kelfi.kernel_means_inference import kernel_means_weights, approximate_marginal_kernel_means_likelihood
from kelfi.kernel_means_inference import kernel_means_posterior, approximate_kernel_means_posterior_embedding, kernel_herding
from kelfi.kernel_means_learning import kernel_means_hyperparameter_learning
from kelfi.kernels import gaussian_kernel_gramix

def hyperparameter_learning_objective(y, x_sim, t_sim, t_samples, beta, eps, reg=None):
    """Computes the approximate MKML for different hyperparameters."""
    weights = kernel_means_weights(y, x_sim, t_sim, eps, beta, reg=reg)
    return approximate_marginal_kernel_means_likelihood(t_samples, t_sim, weights, beta)


def kelfi(y, x_sim, t_sim, t_samples, beta, eps, reg=None, n_samples=1000, beta_query=None):
    """Full KELFI Solution."""
    weights = kernel_means_weights(y, x_sim, t_sim, eps, beta, reg=reg)
    mkml = approximate_marginal_kernel_means_likelihood(t_samples, t_sim, weights, beta)
    if beta_query is None:
        beta_query = beta
    kernel_function = lambda t1, t2: gaussian_kernel_gramix(t1, t2, beta_query)
    kmpe_ = approximate_kernel_means_posterior_embedding(t_samples, t_sim, weights, beta, t_samples, marginal_likelihood=mkml, beta_query=beta_query)
    t_kmpe = kernel_herding(kmpe_, kernel_function, t_samples, n_samples)
    return t_kmpe

seed = 0 
np.random.seed(seed)

exp_names = ['TE1', 'TE2', 'TE3', 'BDM', 'NW']
models = [dgp_funcs.multigaussian(), dgp_funcs.multimodal_logistic(), dgp_funcs.beta_x(),
          bdm_dgp.bdm_simulator(), navworld.navworld_simulator()] # side = 6, test=True ep = 1
noise_var = [[5], [5], [5], [0.5, 0.03, 1, 0.7], 0]
bounds = [{'t1':(0, 100)}, {'t1':(0, 100)}, {'t1':(0, 100)},
          {'R1':(1.01, 12), 'R2': (0.01, 0.4),'burden': (120, 220), 't1':(0.01, 30)},
          {'white':(-20.0, 0.0), 'yellow':(-20.0, 0.0),'red':(-20.0, 0.0),
          'green':(-20.0, 0.0), 'purple':(-20.0, 0.0)}]
par_names = [['t1'], ['t1'], ['t1'], ['R1', 'R2', 'burden', 't1'],
             ['green', 'purple', 'red', 'white', 'yellow']]
true_pars = [ {'t1': 50}, {'t1': 20}, {'t1': 20},
            {'R1': 5.88, 'R2': 0.09, 'burden': 192, 't1': 6.74},
            {"white": 0.0, "yellow": -1.0, "red": -1.0, "green": -5.0, "purple": -10.0}]

exp = sys.argv[1]
it = int(sys.argv[2])
auto_dif = sys.argv[3]
init_ev = 500


output_folder = 'posteriors/' + surrogate + '/'
try:
    os.mkdir(output_folder[:-1])
    print("Directory " , output_folder,  " created") 
except FileExistsError:
    print("Directory " , output_folder,  " already exists")

np.random.seed(it)

import time
for ind in range(0, len(models)):
    if not exp in exp_names[ind]:
        continue
    
    seed = it
    np.random.seed(seed)
    tf.set_random_seed(seed)
    true_theta = np.array([true_pars[ind][par] for par in par_names[ind]])

    if 'TE' in exp_names[ind]:
        y_data = models[ind].func(true_theta)
    elif exp_names[ind] == 'BDM': 
        models[ind].get_model()
        y_data = [0] # models[ind].y0_sum
    elif exp_names[ind] == 'NW': 
        y_data = models[ind].observed_data

    save_dir = output_folder + exp_names[ind] + '/' + surrogate + '-' + exp_names[ind]
    try:
        os.mkdir(output_folder + exp_names[ind])
        print("Directory " , output_folder + exp_names[ind],  " created") 
    except FileExistsError:
        print("Directory " , output_folder + exp_names[ind],  " already exists")

    t0 = time.clock()
    if path.exists(save_dir + 'z' + str(it) + '.dnpz') == False:
        n_sim = init_ev * 2
        n_prior = 10000
        prior_samples = models[ind]
        m = models[ind].get_model(seed_obs=seed)

        outputs = m.parameter_names + ['sim']
        data = m.generate(batch_size=n_sim, outputs=outputs, seed=it) 
        prior = m.generate(batch_size=n_prior, outputs=m.parameter_names, seed=it) 

        param_sample = np.reshape(data[par_names[ind][0]], (-1, 1))
        t_sample = np.reshape(prior[par_names[ind][0]], (-1, 1))
        for par in par_names[ind][1:]:
            temp_sample = np.reshape(data[par], (-1, 1))
            param_sample = np.concatenate((param_sample, temp_sample), axis=1)
            temp_sample = np.reshape(prior[par], (-1, 1))
            t_sample = np.concatenate((t_sample, temp_sample), axis=1)

        y_data = np.array(y_data)
        x_data = np.reshape(data['sim'], (-1, 1, len(y_data)))
        y_data = np.reshape(y_data, (-1, 1, len(y_data)))
        np.savez(save_dir + 'z' + str(it) + '.npz', name1 = x_data, name2 = y_data, name3 = param_sample, name4 = t_sample)
    else:
        loaded_data = np.load(save_dir + 'z' + str(it) + '.npz')
        print('Load!')
        x_data = loaded_data['name1']
        y_data = loaded_data['name2']
        param_sample = loaded_data['name3']
        t_sample = loaded_data['name4']
        # load



    param_sample_mean = np.mean(param_sample, 0)
    param_sample_std = np.std(param_sample, 0)

    param_sample = (param_sample - param_sample_mean) / param_sample_std

    t_sample = (t_sample - param_sample_mean) / param_sample_std

    
    # Full hyperparameter learning with automatic differentiation
    if auto_dif == True:
        eps_tuple = (0.06, 'learn')
        beta_tuple = (0.6, 'learn')
        reg_tuple = (1e-6, 'learn')
        eps_opt, beta_opt, reg_opt = kernel_means_hyperparameter_learning(
            y_data, x_data, param_sample, eps_tuple, beta_tuple, reg_tuple,
            eps_ratios=1., beta_ratios=1., offset=0.,
            prior_samples=t_sample, prior_mean=None, prior_std=None,
            learning_rate=0.01, n_iter=5000, display_steps=100)
    else:
        beta_array = np.linspace(0.5, 1.5, 100)
        eps_array = np.linspace(0.05, 0.15, 100)
        eps_grid, beta_grid = np.meshgrid(eps_array, beta_array)

        mkml_grid = np.zeros((beta_array.shape[0], eps_array.shape[0]))
        mkml_global = -np.inf
        for i, beta in enumerate(beta_array):
            for j, eps in enumerate(eps_array):
                mkml_grid[i, j] = hyperparameter_learning_objective(y_data, x_data, param_sample, t_sample, beta, eps)
                if mkml_grid[i, j] > mkml_global:
                    mkml_global = mkml_grid[i, j]
                    beta_global = beta
                    eps_global = eps

    # calculate the posterior
    n_samples = 100000
    beta_query = 0.1
    t_kmpe_opt = kelfi(y_data, x_data, param_sample, t_sample, beta_global, eps_global, reg=None, n_samples=n_samples, beta_query=beta_query)

    t_kmpe_opt = t_kmpe_opt * param_sample_std + param_sample_mean

    theta_df = pd.DataFrame.from_records(t_kmpe_opt, columns = par_names[ind])
    theta_df.to_pickle(save_dir + '-thetas-' + str(it) + '.pkl')
