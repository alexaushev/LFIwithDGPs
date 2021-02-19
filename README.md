# Likelihood-free Inference with Deep Gaussian Processes

This folder contains all the code required to replicate experiments described in the paper. The simulator that are used in this project are
implemented on top of the Engine for Likelihood-free Inference (ELFI).

## Getting Started

The project dependencies can be installed using the following command in the directory of the project (Python 3.5 version is required):

```
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user -e elfidev/
```

## Running the experiments

There are two script files that need to be run in order to replicate the experiments. The first one is main.py, and it runs the Bayesian Optimization procedure with the chosen surrogate model. It takes 'DGP' (for the DGP-IWVI model), 'GP' (for the vanilla GP model) and
'Rej' (rejection ABC for calculating 'true' posterior) as the main argument. And then, for the first two models the code the name of the experiment ('TE1', 'TE2', 'TE3', 'BDM', 'NW') requires random seeds (1-1000 were used in the paper). The DGP architecture also needs to be specified, it needs 'True' or 'False' for the inclusion of the LV layer, followed by the number of hidden GP layers. The following example runs the DGP model with random seed that equals to 1 and the 'LV-2GP' architecture:

```
python3 main.py DGP BDM 1 True 1
```

Similarly, the GP model and ABC-rejection can be trained as follows:

```
python3 main.py GP BDM 1
python3 main.py Rej
```

Please note, that running all these models with the current configuration (that was used in the paper) require a lot of time: up to 7 hours for the surrogates and up to several days for the rejection ABC. After running the models, it creates files in the 'posterior' folder. The surrogate models are separated in the folders with their names and the result of the rejection ABC is simply put as a file.

Once the posterior samples are ready, the second script 'compare.py' can be run. The script calculates the Wasserstein distance for each surrogate posterior vs 'true' posterior and puts them in a plot for comparison in the 'posteriors/plots/' folder. Since running the surrogate models 1,000 times is a computationally expensive task, we included all files with already calculated Wasserstein distances, so the 'compare.py' script can be run without first executing 'main.py'. Please notice, that if you want to replicate the experiments from the appendix, you need to copy and paste file from 'posteriors/appendix-experiments' to 'posteriors/' and uncomment line 29 in 'compare.py'.

## Running alternative models

MAF, MDNs and KELFI are computed in separate scripts: 'main-snl.py', 'main-kelfi.py'. They can be run similarly as the script from the main experiments. For the 'main-snl.py', the model ('MAF' or 'MDN'), the experiment name and the random seed number are needed. For example:

'''
python3 main-snl.py MAF BDM 1
'''

For 'main-kelfi.py', instead of the name of the model, the automatic differentation for hyperparameter learning should be specified ('True' or 'False'):

'''
python3 main-kelfi.py BDM 1 False
'''

## Citation

'''
@article{aushev2020likelihood,
  title={Likelihood-Free Inference with Deep Gaussian Processes},
  author={Aushev, Alexander and Pesonen, Henri and Heinonen, Markus and Corander, Jukka and Kaski, Samuel},
  journal={arXiv preprint arXiv:2006.10571},
  year={2020}
}
'''

the arxiv version of the paper: https://arxiv.org/pdf/2006.10571.pdf 
