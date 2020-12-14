'''Neural density estimation models are adapted from https://github.com/justinalsing/pydelfi'''

import numpy as np
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.score as score
import pydelfi.priors as priors
import tensorflow as tf
import pandas as pd
import pickle, time, os, sys


from elfi.examples import dgp_funcs, bdm_dgp, navworld

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


surrogate = sys.argv[1]
exp = sys.argv[2]
it = int(sys.argv[3])
init_ev = 500


output_folder = 'posteriors/' + surrogate + '/'
try:
    os.mkdir(output_folder[:-1])
    print("Directory " , output_folder,  " created") 
except FileExistsError:
    print("Directory " , output_folder,  " already exists")

np.random.seed(it)
for ind in range(0, len(models)):

    if not exp in exp_names[ind]:
        continue

    seed = it
    np.random.seed(seed)
    tf.set_random_seed(seed)

    def simulator(theta, seed, simulator_args, batch):
        return models[ind].func(theta, batch_size=batch)
    simulator_args = None

    def compressor(d, compressor_args):
        return d
    compressor_args=None

    # 
    # dict_values([(0.01, 0.4), (0.01, 30), (120, 220), (1.01, 12)])
    # print(bounds[ind].values())
    lower = np.array([bounds[ind][par][0] for par in par_names[ind]] )
    upper = np.array([bounds[ind][par][1] for par in par_names[ind]] )
    prior = priors.Uniform(lower, upper)

    true_theta = np.array([true_pars[ind][par] for par in par_names[ind]])

    if 'TE' in exp_names[ind]:
        compressed_data = models[ind].func(true_theta)
    elif exp_names[ind] == 'BDM': 
        models[ind].get_model()
        compressed_data = models[ind].y0_sum
    elif exp_names[ind] == 'NW': 
        compressed_data = models[ind].observed_data

    theta_dim = len(true_pars[ind].values())
    data_dim = len(compressed_data)

    if surrogate == 'MDN':
        NDEs = [ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=0),
        ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
        ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
        ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
        ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4)]
    elif surrogate == 'MAF':
        NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=theta_dim, n_data=data_dim, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=5)]

    res = simulator(true_theta, seed = it, simulator_args = None, batch = 2)
    print(res)
    DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                                param_limits = [lower, upper],
                                param_names = par_names[ind],
                                show_plot=False, progress_bar=False, save=False, graph_restore_filename='', restore_filename='')
                                # ['\\Omega_m', 'w_0', 'M_\mathrm{B}', '\\alpha', '\\beta', '\\delta M'], 
                                #results_dir = "simulators/jla_supernovae/results/")


    # DelfiEnsemble.fisher_pretraining()
    n_initial = init_ev
    n_batch = 5
    n_populations = int(init_ev / 5)

    DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations, patience=20,
                        save_intermediate_posteriors=False, plot = False)

    save_dir = output_folder + exp_names[ind] + '/' + surrogate + '-' + exp_names[ind]
    try:
        os.mkdir(output_folder + exp_names[ind])
        print("Directory " , output_folder + exp_names[ind],  " created") 
    except FileExistsError:
        print("Directory " , output_folder + exp_names[ind],  " already exists")
    

    posterior_samples = DelfiEnsemble.emcee_sample()
    theta_df = pd.DataFrame.from_records(posterior_samples, columns = par_names[ind])
    theta_df.to_pickle(save_dir + '-thetas-' + str(it) + '.pkl')

    DelfiEnsemble.sequential_training_plot(savefig=True, filename=save_dir + '-training-plot-'
                   + str(it) + '.png')
    DelfiEnsemble.triangle_plot(samples=[posterior_samples], savefig=True, filename=save_dir + '-pair-plot-'
                   + str(it) + '.png')

