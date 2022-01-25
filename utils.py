import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ot, scipy

import matplotlib as mpl

import numpy as np
import pandas as pd
from os import path
from pathlib import Path

from kelfi.kernel_means_inference import kernel_means_weights, approximate_marginal_kernel_means_likelihood
from kelfi.kernel_means_inference import approximate_kernel_means_posterior_embedding, kernel_herding
from kelfi.kernel_means_learning import kernel_means_hyperparameter_learning
from kelfi.kernels import gaussian_kernel_gramix


import warnings
warnings.filterwarnings("ignore")


# =====================
# PLOTTING FUNCTIONS:
# =====================

mpl.rcParams['figure.dpi'] = 300


# sample posterior of the trained surrogate model
def get_weighted_samples(post, N=100000):
    theta = post.prior.rvs(size=N)

    if theta.ndim == 1:
        theta = theta.reshape(theta.shape[0], 1)
        
    weights = post._unnormalized_likelihood(theta)
    return theta, weights



def sample_posterior(samples, weights, cols, N=100000):
    theta = samples
    n_weights = weights / np.sum(weights)

    # importance weighted resampling
    resample_index = np.random.choice(len(samples), size=N, replace=True, p=n_weights)
    theta_resampled = theta[resample_index,:]
    theta_df = pd.DataFrame.from_records(theta_resampled, columns = cols)
    return theta_df



# plot the grid plot of the surrogate posterior
def plot_grid(theta_df, lims):
    g = sns.PairGrid(theta_df)
    for x_ind in range(0, len(lims)):
        for y_ind in range(0, len(lims)):
            g.axes[y_ind, x_ind].set_xlim(lims[x_ind])
            g.axes[y_ind, x_ind].set_ylim(lims[y_ind])

    g = g.map_lower(plt.scatter, s=1)
    g = g.map_diag(plt.hist)


def plot_thresholds(thrs, legend):
    ax = plt.gca()
    ax.plot(thrs)
    ax.set_ylabel('threshold')
    ax.set_xlabel('num of batches')
    ax.legend(legend)
    return



def plot_marginals(sim, bounds, true_samples_df, true_pars):
    plot_methods = ['BO-GP(200)', 'BO-LV-2GP-0.3(200)']
    par_names = list(bounds.keys())
    sns.set(style="ticks", rc={"lines.linewidth": 0.7})

    maps = {'t1': r'$\theta_{t1}$', 'R1': r'$\theta_{R1}$', 'R2': r'$\theta_{R2}$',
        'burden': r'$\theta_{burden}$', 'white': r'$\theta_{white}$',
        'yellow': r'$\theta_{yellow}$', 'red': r'$\theta_{red}$',
        'green': r'$\theta_{green}$', 'purple': r'$\theta_{purple}$',
        'x': r'$\theta_{x}$', 'y': r'$\theta_{y}$', 
        'ns': r'$\theta_{n_s}$', 'kc': r'$\theta_{k_c}$', 'alpha': r'$\theta_{\alpha}$', 
        'r_star': r'$\theta_{R^*}$', 'As': r'$\theta_{A_s}$'}

    for plot_method in plot_methods:
        for cur_par_name, i in zip(par_names, range(len(par_names))):
            fig = plt.gcf()
            for filename in Path('./results/' + sim + '/' + plot_method).glob('*.samples'):
                temp_dict = scipy.io.loadmat(filename)
                sur_theta = temp_dict[cur_par_name]
                # sns.distplot(sur_theta, color=(0.879, 0.929, 0.969), hist=False, kde_kws={'alpha':0.1}) #, scatter_kws={'alpha':0.3})
                # sns.distplot(sur_theta, color=(0.711, 0.832, 0.91), hist=False, kde_kws={'alpha':0.05}) # DGP almost good
                # sns.distplot(sur_theta, color=(0.554, 0.734, 0.855), hist=False, kde_kws={'alpha':0.05}) #GP gppd
                try:
                    sns.distplot(sur_theta, color=(0.516, 0.707, 0.839), hist=False, kde_kws={'alpha':0.1})
                except np.linalg.LinAlgError:
                    sns.distplot(sur_theta, color=(0.516, 0.707, 0.839), kde=False, kde_kws={'alpha':0.1})

            true_theta = pd.Series(true_samples_df[cur_par_name], name = cur_par_name)
            sns.distplot(true_theta, color="r", hist=False, kde_kws={"linewidth": 1})
            plt.axvline(true_pars[i], ls = 'dashed', color="black")
            plt.yticks([])

            if 'TE' in sim:
                plt.xlabel(r'$\theta$')
            else:
                plt.xlabel(maps[cur_par_name])
            plt.ylabel('')
            # plt.ylim((0,2.5))

            # fig.set_size_inches(2.5,2.1) # in TE plots
            # fig.set_size_inches(1.4,1.4) # in 5 row plots
            fig.set_size_inches(1.8,1.8)
            plt.savefig('plots/' + sim + '-' + plot_method + '-par-' +  cur_par_name + '.png', dpi=600, bbox_inches = 'tight')
            plt.close()



def plot_wasserstein(boxplot=True):
    plot_methods = ['BO-GP', 'BO-LV-2GP-0.3']

    sns.set(style="ticks", rc={"lines.linewidth": 0.7})
    datasets = ['20', '50', '100', '150', '200', '250', '300']

    p = Path('./results')
        
    # All subdirectories in the current directory, not recursive.
    sim_paths = [f for f in p.iterdir() if f.is_dir()]
        
    for sim_path in sim_paths: 
        sim = sim_path.name.split('/')[-1]
        wass_dist_df = pd.DataFrame(columns=['Model', 'dset', 'wass'])
        meth_paths = [f for f in sim_path.iterdir() if f.is_dir()]            
        for meth_path in meth_paths:
            if plot_methods[1] in meth_path.name:
                plot_method = 'LV-2GP'
            elif plot_methods[0] in meth_path.name:
                plot_method = 'GP'
            else:
                continue
    
            dset = meth_path.name.split('(')[-1][:-1]
            if boxplot is False and int(dset) != 200:
                continue
            
            for filename in Path(meth_path).glob('*.mat'):	
                temp_dict = scipy.io.loadmat(filename)

                if temp_dict['Wass'][0][0] < 0:
                    print(filename, temp_dict['Wass'][0][0])
                    continue
                data = {'Model': plot_method, 'dset': dset, 'wass': temp_dict['Wass'][0][0]}
                wass_dist_df = wass_dist_df.append(data, ignore_index=True)

        if wass_dist_df.empty:
            continue
        fig = plt.gcf()
        my_pal = {m: "r" if m == "GP" else "b" for m in wass_dist_df.Model.unique()}

        if boxplot is True:    
            ax = sns.boxplot(x='dset', y='wass', hue='Model', data=wass_dist_df, linewidth=2, \
                palette=my_pal, showfliers = False, order=['20','50','100','150','200','250','300'])
            plt.legend([],[], frameon=False)
            plt.xlabel(r'Number of simulations')
            fig.set_size_inches(5,3)
            plt.ylabel(r'',rotation=0)
        else:
            ax = sns.violinplot(x='Model', y='wass', data=wass_dist_df, linewidth=1.1, palette=my_pal, cut=0)
            plt.xlabel(r'')
            fig.set_size_inches(1.5,2.2)

            plt.ylabel(r'',rotation=0)
        
        plt.savefig('plots/' + sim + '-wass.png', dpi=600, bbox_inches = 'tight')
        plt.close()

# =====================
# KELFI FUNCTIONS:
# =====================

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



def generate_data(elfi_model, y_data, n_sim, n_prior_samples, par_names, seed):
    '''Generates the data from the ELFI model, which will be later used for training'''
    outputs = elfi_model.parameter_names + ['sim']
    data = elfi_model.generate(batch_size=n_sim, outputs=outputs, seed=seed) 
    prior = elfi_model.generate(batch_size=n_prior_samples, outputs=par_names, seed=seed) 

    parameters = np.reshape(data[par_names[0]], (-1, 1))
    pars_from_prior = np.reshape(prior[par_names[0]], (-1, 1))
    for par in par_names[1:]:
        temp_parameters = np.reshape(data[par], (-1, 1))
        parameters = np.concatenate((parameters, temp_parameters), axis=1)
        temp_pars_from_prior = np.reshape(prior[par], (-1, 1))
        pars_from_prior = np.concatenate((pars_from_prior, temp_pars_from_prior), axis=1)

    y_data = np.array(y_data)
    x_data = np.reshape(data['sim'], (-1, 1, len(y_data)))
    y_data = np.reshape(y_data, (-1, 1, len(y_data)))
    return x_data, y_data, parameters, pars_from_prior


def tune_hyperparameters(x_data, y_data, parameters, pars_from_prior, auto_dif=True):
    if auto_dif == True:
        eps_tuple = (0.06, 'learn')
        beta_tuple = (0.6, 'learn')
        reg_tuple = (1e-6, 'learn')
        eps, beta, reg_opt = kernel_means_hyperparameter_learning(
            y_data, x_data, parameters, eps_tuple, beta_tuple, reg_tuple,
            eps_ratios=1., beta_ratios=1., offset=0.,
            prior_samples=pars_from_prior, prior_mean=None, prior_std=None,
            learning_rate=0.01, n_iter=5000, display_steps=100)
    else:
        beta_array = np.linspace(0.5, 1.5, 100)
        eps_array = np.linspace(0.05, 0.15, 100)

        mkml_grid = np.zeros((beta_array.shape[0], eps_array.shape[0]))
        mkml_global = -np.inf
        for i, beta in enumerate(beta_array):
            for j, eps in enumerate(eps_array):
                mkml_grid[i, j] = hyperparameter_learning_objective(y_data, x_data, parameters, pars_from_prior, beta, eps)
                if mkml_grid[i, j] > mkml_global:
                    mkml_global = mkml_grid[i, j]
                    beta = beta
                    eps = eps

    return beta, eps



# =====================
# WASSERSTEIN DISTANCE:
# =====================


def get_wass_dist(samples_1, samples_2, weights_1=None, weights_2=None, num_iter_max=1000000, **kwargs):
    """
    Computes the Wasserstein 2 distance between two empirical distributions with weights. This uses the POT library to 
    estimate Wasserstein distance. The Wasserstein distance computation can take long if the number of samples in the 
    two datasets is large (cost of the computation scales in fact quadratically with the number of samples).
    Parameters
    ----------
    samples_1 : np.ndarray
         Samples defining the first empirical distribution, with shape (nxd), n being the number of samples in the
         first empirical distribution and d the dimension of the random variable.
    samples_2 : np.ndarray
         Samples defining the second empirical distribution, with shape (mxd), m being the number of samples in the
         second empirical distribution and d the dimension of the random variable.
    weights_1 : np.ndarray, optional
         Weights defining the first empirical distribution, with shape (n), n being the number of samples in the
         first empirical distribution. Weights are normalized internally to the function. If not provided, they are
         assumed to be identical for all samples.
    weights_2 : np.ndarray, optional
         Weights defining the second empirical distribution, with shape (m), m being the number of samples in the
         second empirical distribution. Weights are normalized internally to the function. If not provided, they are
         assumed to be identical for all samples.
    num_iter_max : integer, optional
        The maximum number of iterations in the linear programming algorithm to estimate the Wasserstein distance. 
        Default to 100000. 
    kwargs 
        Additional arguments passed to ot.emd2
    Returns
    -------
    float
        The estimated 2-Wasserstein distance.
    """
    n = samples_1.shape[0]
    m = samples_2.shape[0]

    if weights_1 is None:
        a = np.ones((n,)) / n
    else:
        if len(weights_1) != n:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        a = weights_1 / np.sum(weights_1)
    if weights_2 is None:
        b = np.ones((m,)) / m
    else:
        if len(weights_2) != m:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        b = weights_2 / np.sum(weights_2)

    # loss matrix
    M = ot.dist(x1=samples_1, x2=samples_2)  # this returns squared distance!
    cost = ot.emd2(a, b, M, numItermax=num_iter_max, **kwargs)

    return np.sqrt(cost)




