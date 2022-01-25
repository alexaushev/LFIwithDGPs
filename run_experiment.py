import argparse, sys
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import time, scipy
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import elfi
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.dgp_regression import DGPRegression
from elfi.methods.bo.acquisition import LCBSC
from elfi.examples import dgp_funcs, bdm_dgp, navworld, sound_loc, cosm_inflation

import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.priors as priors

from utils import sample_posterior, get_weighted_samples, plot_grid, kelfi, generate_data, \
    tune_hyperparameters, get_wass_dist, plot_marginals, plot_wasserstein
    

if __name__ == "__main__":
    # parse arguments of the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim')
    parser.add_argument('--meth')
    parser.add_argument('--surrogate')
    parser.add_argument('--seed')
    parser.add_argument('--gplayers')
    parser.add_argument('--lv')
    parser.add_argument('--q')
    parser.add_argument('--evidence')
    
    args = parser.parse_args()
    sim = str(args.sim)
    meth = str(args.meth)

    start = time.time()

    # fix random seeds
    seed = int(args.seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # ==========================
    # SET SIMULATOR:
    # ==========================
    if sim == 'TE1':
        sim_model = dgp_funcs.multigaussian()
        noise_var = [5]
        bounds = {'t1':(0, 100)}
        true_pars = np.array([50])
        compressed_data = sim_model.func(true_pars)

    elif sim == 'TE2':
        sim_model = dgp_funcs.multimodal_logistic(n=0.5, offset=50)
        noise_var = [5]
        bounds = {'t1':(0, 100)}
        true_pars = np.array([20])
        compressed_data = sim_model.func(true_pars)
     
    elif sim == 'TE3':
        sim_model = dgp_funcs.beta_x()
        noise_var = [5]
        bounds = {'t1':(0, 100)}
        true_pars = np.array([20])
        compressed_data = sim_model.func(true_pars)

    elif sim == 'TE4':
        sim_model = dgp_funcs.bigaussian() # multimodal_logistic(n=0.2, offset=70, noise=0.05)
        noise_var = [5]
        bounds = {'t1':(0, 100)}
        true_pars = np.array([50])
        # dgp_funcs.plot_d(sim_model.func, true_pars, bounds)
        compressed_data = sim_model.func(true_pars)
        
    elif sim == 'BDM':
        sim_model = bdm_dgp.bdm_simulator()
        noise_var = [0.5, 0.03, 1, 0.7]
        bounds = {'R1':(1.01, 12), 'R2': (0.01, 0.4),'burden': (120, 220), 
           't1':(0.01, 30)}
        true_pars = np.array([5.88, 0.09, 192, 6.74])
        compressed_data = [0.]
    
    elif sim == 'NW':
        sim_model = navworld.navworld_simulator()
        noise_var = 0
        bounds = {'white':(-20.0, 0.0), 'yellow':(-20.0, 0.0),'red':(-20.0, 0.0),
          'green':(-20.0, 0.0), 'purple':(-20.0, 0.0)}
        true_pars = np.array([0.0, -1.0, -1.0, -5.0, -10.0])
        compressed_data = sim_model.observed_data

    elif sim == 'SL':
        sim_model = sound_loc.sound_localization()
        noise_var = [0.1, 0.1]
        bounds = {'x': (-2, 2), 'y':(-2, 2) }
        true_pars = np.array([1.5, 1])
        compressed_data = sim_model.func(true_pars)[0]

    elif sim == 'CI':
        sim_model = cosm_inflation.cosmological_inflation()
        noise_var = [0.05, 0.00005, 0.5, 0.05, 0.05]
        bounds = {'ns': (0.5, 1.5), 'kc':(0, 0.003), 'alpha':(0., 10.),\
            'r_star':(0., 1.), 'As': (2.7, 4.) }
        true_pars = np.array([0.96, 0.0003, 0.58, 0.75, 3.35 ])
        compressed_data = sim_model.func(true_pars)

    elfi_model = sim_model.get_model(seed_obs=seed)
    par_names = list(bounds.keys())
    par_names.sort()

    inference_flag = meth != 'Rej' and meth != 'True'

    # fetch true posterior if available
    true_samples_filename = 'results/' +  sim + '/True/true_samples.mat'
    true_samples_file = Path(true_samples_filename)
    if not(true_samples_file.is_file()) and inference_flag:
        raise ValueError("Calculate true posterior with --meth=Rej and then --meth=True")
    elif true_samples_file.is_file():
        true_samples_dict = scipy.io.loadmat(true_samples_filename)
        true_samples_dict = { key: true_samples_dict[key].flatten() for key in par_names }
        true_samples_df = pd.DataFrame(true_samples_dict)        
    
    # ==========================
    # DO INFERENCE:
    # ========================== 
    evidence = int(args.evidence)
    if meth == 'BO':
        surrogate = str(args.surrogate)
        q = float(args.q)
        init_ev = int(evidence / 2)
        if surrogate == 'GP':
            target_model = GPyRegression(parameter_names=par_names, bounds=bounds)
            meth += '-' + surrogate
        elif surrogate == 'DGP':
            # set the DGP architecture
            LVlayer = eval(args.lv)
            GPlayers = int(args.gplayers)-1 if LVlayer is True else int(args.gplayers)-2
            surrogate = 'LV-' if LVlayer else ''
            surrogate += str(GPlayers + 1) + '*GP'
            target_model = DGPRegression(parameter_names=par_names, bounds=bounds, GPlayers=GPlayers, LVlayer=LVlayer,
                                        Ms=50, IW_samples=5, pred_samples=20, opt_steps=20000, q=q)
            meth += '-' + surrogate
            meth += '-' + str(q)
        meth += '(' + str(evidence) + ')'
        acq =  LCBSC(target_model, noise_var=noise_var, exploration_rate=10, seed=seed)
        bolfi = elfi.BOLFI(elfi_model, 'dist', batch_size=5, initial_evidence=init_ev, update_interval=init_ev, 
                            target_model = target_model, acquisition_method = acq, seed=seed)
        # conduct inference
        post, mses = bolfi.fit(n_evidence=evidence, bar=False)
        res = bolfi.extract_result()
        samples, weights = get_weighted_samples(post, N=100000)
        samples_df = None
        # posterior_samples = sample_posterior(samples, weights, cols=par_names, N=10000)

    elif meth == 'NDE':
        surrogate = str(args.surrogate)
        theta_dim = len(par_names)
        data_dim = len(compressed_data)
        if surrogate == 'MDN':
            NDEs = [ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=0),
            ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
            ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
            ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
            ndes.MixtureDensityNetwork(n_parameters=theta_dim, n_data=data_dim, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4)]
        elif surrogate == 'MAF':
            NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=theta_dim, n_data=data_dim, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=5)]
        meth += '-' + surrogate
        meth += '(' + str(evidence) + ')'

        # set the prior for the neural surrogate
        lower = np.array([bounds[par][0] for par in par_names] )
        upper = np.array([bounds[par][1] for par in par_names] )
        prior = priors.Uniform(lower, upper)
        restore_file = 'restore/' + sim + '-' + meth + str(seed) 
        DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, param_limits=[lower, upper], param_names=par_names, 
                            show_plot=False, progress_bar=False, save=False, graph_restore_filename=restore_file, restore_filename=restore_file)
        n_batch = 5
        n_populations = int(evidence / 5)
        # these two functions are required by the API
        def simulator(theta, seed, simulator_args, batch): return sim_model.func(theta, batch_size=batch)
        def compressor(d, compressor_args): return d
        # conduct inference
        DelfiEnsemble.sequential_training(simulator, compressor, evidence, n_batch, n_populations, patience=20,
                        save_intermediate_posteriors=False, plot = False)
        samples, weights, _ = DelfiEnsemble.emcee_sample()
        # n_weights = weights / np.sum(weights)
        # resample_index = np.random.choice(len(posterior_samples), size=10000, replace=True, p=n_weights)
        samples_df = None

    elif meth == 'KELFI':
        # autodif = bool(args.autodif)
        # generate data
        x_data, y_data, parameters, pars_from_prior = generate_data(elfi_model, compressed_data, n_sim=evidence, \
            n_prior_samples=10000, par_names=par_names, seed=seed)
        par_mean, par_std = np.mean(parameters, 0), np.std(parameters, 0)
        # normalize parameters for training
        parameters = (parameters - par_mean) / par_std
        pars_from_prior = (pars_from_prior - par_mean) / par_std
        # search for hyperparameters (optimize with auto_dif=True)
        beta, eps = tune_hyperparameters(x_data, y_data, parameters, pars_from_prior, auto_dif=False)
        # conduct inference
        n_samples = 10000
        beta_query = 0.1
        t_kmpe_opt = kelfi(y_data, x_data, parameters, pars_from_prior, beta, eps, reg=None, n_samples=n_samples, beta_query=beta_query)
        # unnormalize parameters before saving
        t_kmpe_opt = t_kmpe_opt * par_std + par_mean
        samples = t_kmpe_opt # = pd.DataFrame.from_records(t_kmpe_opt, columns=par_names)
        weights = None
        samples_df = pd.DataFrame.from_records(t_kmpe_opt, columns=par_names)
        meth += '(' + str(evidence) + ')'

    elif meth == 'Rej':
        n_sim = evidence # 10e7 10e3
        n_samples = evidence # 10e5 10e2
        rej = elfi.Rejection(elfi_model, 'dist', seed=seed, batch_size=1, max_parallel_batches=16)
        rej.set_objective(n_samples=n_samples, n_sim=n_sim)
        while not rej.finished:
            rej.iterate()
        # res = rej.extract_result()
        #print(rej.state['samples'])
        samples_df = pd.DataFrame.from_records(rej.state['samples'])
        print(samples_df)

    elif meth == 'True': 
        # collect all rejection samples in one dataframe
        paths = Path('results/' +  sim + '/Rej/').glob('*.mat')
        column_names = par_names + ['dist']

        true_samples = []
        for filename in paths:
            temp_dict = scipy.io.loadmat(filename)
            temp_dict = { key: temp_dict[key].flatten() for key in column_names }
            temp_df = pd.DataFrame(temp_dict)
            true_samples.append(temp_df)

        if not true_samples:
            raise ValueError('Generate samples with --meth=Rej first')
        else:
            true_samples_df = pd.concat(true_samples)

        # leave only 10 000 samples with the lowest discrepancy
        true_samples_df = true_samples_df.sort_values(by=['dist']) 
        samples_df = true_samples_df.head(evidence)
        samples_df = samples_df.drop(['dist'], axis=1)
        print(samples_df)

    elif meth == 'Tex':
        p = Path('./results')

        # All subdirectories in the current directory, not recursive.
        sim_paths = [f for f in p.iterdir() if f.is_dir()]

        for sim_path in sorted(sim_paths):
            print('\n', sim_path.name)
            meth_paths = [f for f in sim_path.iterdir() if f.is_dir()]
            
            for meth_path in sorted(meth_paths):
                
                # do not show results for these folders            
                if 'Rej' in meth_path.name or 'True' in meth_path.name: 
                    continue
                
                # epsilon_q tables:    
                if 'LV-2GP' not in meth_path.name or '200' not in meth_path.name:
                    continue
                
                #if '(200)' not in meth_path.name and '(1000)' not in meth_path.name:
                #    continue

                #if 'LV-2GP' in meth_path.name and '0.3' not in meth_path.name:
                #    continue
		
                print(meth_path.name)
                est_times = list()
                est_wass_dist = list()
                for filename in Path(meth_path).glob('*.mat'):
                    temp_dict = scipy.io.loadmat(filename)
                    if temp_dict['Wass'][0][0] > 10e7:
                        continue
                    est_times.append( temp_dict['Time'])
                    est_wass_dist.append( temp_dict['Wass'])
                print('Time: ${:.2f}'.format(np.mean(est_times)), ' \pm {:.2f}'.format(np.std(est_times)), '$')
                print('Wass: ${:.2f}'.format(np.mean(est_wass_dist)), ' \pm {:.2f}'.format(np.std(est_wass_dist)), '$')
                print('Count:', len(est_times))
        sys.exit()

    elif meth == 'MargPlot':
        plot_marginals(sim, bounds, true_samples_df, true_pars)
        sys.exit()

    elif meth == 'WassPlot':
        plot_wasserstein(boxplot=False)
        sys.exit()

    # measure the empirical time
    end_time = (time.time() - start) / 60.

    # calculate the distance
    if inference_flag:
        wass_dist = get_wass_dist(true_samples_df.to_numpy(), samples, weights_2=weights)
        
        # get posterior samples for plotting
        if samples_df is None:
            samples_df = sample_posterior(samples, weights, cols=par_names, N=10000)

    # ==========================
    # PLOT SAMPLES:
    # ==========================
    output_folder = 'results/' +  sim + '/' + meth + '/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if seed >= 0 and seed < 10:
        if meth == 'True':
            seed = 'true_samples'

        samples_df = samples_df[~samples_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        plot_grid(samples_df, lims = [list(bounds[key]) for key in par_names])
        plt.savefig(output_folder + str(seed) + '-plot.png', dpi=300)
        plt.close()

        if 'TE' in sim and str(args.meth)=='BO':
            dgp_funcs.plot_posterior_samples(target_model, x_counts=1000, samples=100, points = True)
            plt.savefig(output_folder + str(seed) + '-state.png', dpi=300)
            plt.close()
    
    # ==========================
    # STORE RESULTS:
    # ==========================
    # save Wasserstein distance and empirical time as results
    save_file_name = output_folder + str(seed) + '.mat'
    print(save_file_name)
    mdict = samples_df.to_dict('list') if meth == 'Rej' or meth == 'True' else {'Wass': wass_dist, 'Time': end_time}
    scipy.io.savemat(save_file_name, mdict=mdict)

    if meth == 'BO-GP(200)' or meth == 'BO-LV-2GP-0.3(200)':
        save_file_name = output_folder + str(seed) + '.samples'
        mdict = samples_df.to_dict('list')
        scipy.io.savemat(save_file_name, mdict=mdict)


    
