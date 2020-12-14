import pandas as pd
import pickle, time, os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import glob, time
import numpy as np
import tensorflow as tf
import scipy

from wasserstein import Wasserstein
from dataset import Dataset

import sys
sys.path.insert(1,'/u/39/ausheva1/unix/Downloads/ite/')


tf.flags.DEFINE_string('filepattern', '/tmp/cifar10/cifar_train_class_%d.pic',
                       'Filepattern from which to read the dataset.')
tf.flags.DEFINE_integer('batch_size', 1000, 'Batch size of generator.')
tf.flags.DEFINE_integer('loss_steps', 50, 'Number of optimization steps.')

FLAGS = tf.flags.FLAGS


names = ['TE1', 'TE2', 'TE3', 'BDM', 'NW']

surs = ['LV-1GP',  'GP']
# all experiments (including the appendix)
# surs = ['1GP', '3GP', 'LV-1GP', 'LV-2GP', 'LV-3GP', 'LV-4GP', 'LV-5GP', 'GP', 'MAF', 'MDN', 'KELFI']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


post_folder = 'posteriors/'
plot_folder = 'posteriors/plots/'
bplot = dict()

# calculate or load wasserstein distance
for sur in surs:
    print('\n===== ' + sur + ' =====')
    bplot[sur] = dict()

    for name in names:
        
        samples_file = open(post_folder + 'Rej-' + name + '-true-posterior-samples.pkl', 'rb')
        true_samples = pickle.load(samples_file)
        theta_names = list(true_samples.columns)
        true_samples = true_samples.to_numpy()
        num_thetas = len(true_samples[0])
        dist_thetas = {key: list() for key in theta_names}
        mean_thetas = {key: list() for key in theta_names}
   
        sur_files = glob.glob(post_folder + sur + '/' + sur + '-' + name + '-thetas-*')
        if os.path.isfile(post_folder + sur + '-' + name + 'Wf.pkl'):
            with open(post_folder + sur + '-' + name + 'Wf.pkl', 'rb') as f:
                bplot[sur][name] = pickle.load(f)
            continue

        print(name)
        bplot[sur][name] = list()
        for file_name in sur_files:            
            sur_file = open(file_name, 'rb')
            try:
                sur_sample = pickle.load(sur_file)
            except TypeError:
                print('TypeError: ' + sur_file)
                continue
            
            sur_sample = sur_sample.to_numpy()
            start_time = time.time()
            
            x1 = Dataset(bs=FLAGS.batch_size, filepattern=true_samples)
            x2 = Dataset(bs=FLAGS.batch_size, filepattern=sur_sample)

            with tf.Graph().as_default():
                wasserstein = Wasserstein(x1, x2)
                loss = wasserstein.dist(C=.1, nsteps=50)
                with tf.Session() as sess:
                  sess.run(tf.global_variables_initializer())
                  res = sess.run(loss)

            bplot[sur][name].append(res)

        f = open(post_folder + sur + '-' + name + 'Wf.pkl', 'wb')
        pickle.dump(bplot[sur][name], f)
        f.close()


# plot wasserstein distance
for name in names:
    cur_plot = dict()
    print(name)
    for sur in surs: 
        cur_plot[sur] = (bplot[sur][name])
        
    cur_plot = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in cur_plot.items()]))
    cur_plot = cur_plot.div(np.min(cur_plot.values))

    for sur in surs:
        pr = 2
        print(sur)
        mean, ci_low, ci_high = mean_confidence_interval(cur_plot[sur])
        print( 'Mean:' + str(round(mean, pr)) + ' CI: ('
               + str(round(ci_low, pr)) + ', ' + str(round(ci_high, pr)) + ')')

    fig = plt.gcf()
    if name == 'BDM':
        cur_plot = np.log(cur_plot)
        plt.ylabel(r'$log(W_D(p_{sur}(\theta), p(\theta)))$')
    else:
        plt.ylabel(r'$W_D(p_{sur}(\theta), p(\theta))$')
        
    fig.set_size_inches(6,4)
    sns.violinplot(data = cur_plot, width = 0.9, cut=0.9)
    plt.title(name)
    plt.savefig(plot_folder + name, dpi=300)
    plt.close()

