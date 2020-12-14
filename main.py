import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle, time, os, sys

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.dgp_regression import DGPRegression
from elfi.methods.bo.acquisition import LCBSC, QuantileAcquisition
from elfi.visualization.interactive import plot_func

import matplotlib as mpl
import importlib
import elfi
import dill as pickle
from elfi.examples import dgp_funcs, bdm_dgp, navworld

# sample posterior of the trained surrogate model
def sample_posterior(post, cols, N=100000):
    theta = post.prior.rvs(size=N)

    if theta.ndim == 1:
        theta = theta.reshape(theta.shape[0], 1)
        
    weights = post._unnormalized_likelihood(theta)
    n_weights = weights / np.sum(weights)
    sample_mean = np.dot(n_weights,theta)

    # importance weighted resampling
    resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
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

mpl.rcParams['figure.dpi'] = 300

seed = 0 
np.random.seed(seed)
tf.set_random_seed(seed)

if sys.argv[1] == 'GP' or sys.argv[1] == 'DGP':
    surrogate = 'GP'
    it = int(sys.argv[2]) # the seed of the experiment

    if sys.argv[1] == 'DGP':
        LVlayer = bool(sys.argv[4])
        GPlayers = int(sys.argv[5])

        if LVlayer:
            surrogate = 'LV-'
        else:
            surrogate = ''
    
        surrogate += str(GPlayers) + 'GP'
    inference = 'BO'
else:
    inference = sys.argv[1]
exp = sys.argv[3]




init_ev = 100 # initial evidence 100
steps =  20000 # in the paper we used 20000 optimiaztion steps
output_folder = 'posteriors/' + surrogate + '/'


try:
    os.mkdir(output_folder[:-1])
    print("Directory " , output_folder,  " created") 
except FileExistsError:
    print("Directory " , output_folder,  " already exists")

# Setup the parallel client
# The actual study used ipyparallel
# elfi.set_client('multiprocessing')

# === INITIALIZATION:
# + exp_names   -- names of the experiments (used when creating files);
# + models      -- all simulators that are used in the experiments;
# + nois_var    -- noise variance that is added to the acquisition function output;
# + bounds      -- parameter bounds for each simulator with parameter names;
# + par_names   -- names of each simulator parameter (this list insures that the correct
#               order is used everywhere in the experiments);

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


# === INFERENCE:
if inference == 'BO':
    print('Bayesian Optimization inference:')
    print(surrogate)
    for ind in range(0, len(models)):
        
        if not exp in exp_names[ind]:
            continue
        
        print(exp_names[ind])
        seed = it
        np.random.seed(seed)
        tf.set_random_seed(seed)

        m = models[ind].get_model(seed_obs=seed)
        start_time = time.time()
        
        if surrogate == 'GP':
            target_model = GPyRegression(parameter_names=par_names[ind],
                                         bounds = bounds[ind])
            acq =  LCBSC(target_model, noise_var=noise_var[ind], exploration_rate=10,
                         seed=seed)
        else:
            target_model = DGPRegression(parameter_names=par_names[ind],
                                         bounds = bounds[ind], layers=GPlayers, Ms = 50,
                                         IW_samples = 5, pred_samples = 20,
                                         opt_steps = steps, q = 0.3)
            acq =  LCBSC(target_model, noise_var=noise_var[ind],
                         exploration_rate=10, seed=seed)
                         
        bolfi = elfi.BOLFI(m, 'dist', batch_size=5, initial_evidence=init_ev,
                           update_interval=init_ev, target_model = target_model,
                           acquisition_method = acq, seed=seed)

        

        post, mses = bolfi.fit(n_evidence=init_ev*2, bar=False) # 100*2

        '''seed = it
        np.random.seed(seed)
        tf.set_random_seed(seed)
        samples = target_model.get_HMC_samples()
        samples_save = list()

        for i in samples:
            for key, value in i.items():
                samples_save.append(value) 

        with open('outfile' + str(it) + exp_names[ind], 'wb') as fp:
            pickle.dump(samples_save, fp)

        raise ValueError'''
        res = bolfi.extract_result()
        print('Time: ' + exp_names[ind] + ' ' +  str(time.time() - start_time))
        raise ValueError

        '''Xs = np.random.uniform(0, 100, 1000)
        target_model.S = 100
        Zs = target_model.sample_fs(Xs)
        
        x_new = np.repeat(Xs, target_model.S)
        plot = sns.jointplot(x=x_new.flatten(), y=Zs.flatten(), kind='kde')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$d(x_{\theta}, x_{obs})$')
        plt.title(r'$f_{d | \theta}(d(x_{\theta}, x_{obs})$')
        plt.savefig('LV-GP-layer.png', dpi=300, bbox_inches = 'tight')'''
        

        theta_df = sample_posterior(post, cols = par_names[ind])
        theta_df.to_pickle(output_folder + surrogate + '-' + exp_names[ind] + '-thetas-'
                          + str(it) + '.pkl')
        plot_grid(theta_df, lims = [list(bounds[ind][key]) for key in par_names[ind]])
        plt.savefig(output_folder + surrogate + '-' + exp_names[ind] + '-pair-plot-'
                   + str(it) + '.png', dpi=100)
        plt.close()

        if 'TE' in exp_names[ind]:
            mpl.rcParams['figure.dpi'] = 300
            xcounts = 1000
            samples = 100
            dgp_funcs.plot_posterior_samples(target_model, x_counts = xcounts, samples = samples, points = True)
            plt.savefig('DGP-state-' + str(it) + '.png', dpi=300)
            plt.close()
        elif exp_names[ind] == 'NW':
            reward = navworld.plot_parameters(res.x_min, models[ind])
            plt.savefig(output_folder + surrogate + '-' + exp_names[ind] + '-map-'
                        + str(it) + '.png', dpi=100)
            plt.close()
            # perf.append(reward)

elif inference == 'Rej':
    print('Rejection inference:')
    n_sim = 100 # 10e7 10e3
    n_samples = 100 # 10e5 10e2

    for ind in range(0, len(models)):     
        m = models[ind].get_model(seed_obs=seed)
        start_time = time.time()
        rej = elfi.Rejection(m, 'dist', seed=seed, batch_size=1, max_parallel_batches=16)
        rej.set_objective(n_samples=n_samples, n_sim=n_sim)
        thrs = []
        i = 0
        while not rej.finished:
            rej.iterate()
            thrs.append([rej.state['samples']['dist'][ss] for ss in [0, 99]])

        plot_thresholds(thrs, ['1', '100', '1000'])
        plt.savefig(output_folder + 'Rej-' + exp_names[ind] + '-treshold-plot.png', dpi=100)
        plt.close()
        res = rej.extract_result()

        samples_df = pd.DataFrame.from_records(res.samples)
        samples_df.to_pickle(output_folder + 'Rej-' + exp_names[ind]
                             + '-true-posterior-samples.pkl')
        plot_grid(samples_df, lims = [list(bounds[ind][key]) for key in par_names[ind]])
        plt.savefig(output_folder + 'Rej-' + exp_names[ind]
                    + '-true-pair-plot.png', dpi=100)
        plt.close()

        times=time.time() - start_time
        f = open(output_folder + + exp_names[ind] + '-Rej-time.pkl', 'wb')
        pickle.dump(times, f)
        f.close()

    
