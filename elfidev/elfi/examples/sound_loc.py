
from functools import partial
import elfi
import numpy as np

from scipy.spatial import distance
from scipy.stats import t
import random


class sound_localization:

    def __init__(self) -> None:
        # locations of two pairs of microphones
        self.mic_pair_1 = [(-0.5, 0), (0.5, 0)] 
        self.mic_pair_2 = [(0, -0.5), (0, 0.5)]
        self.param_dim = 2
        pass


    def func(self, *params, n_obs=100, batch_size=1, random_state=None):
        results = list()
        params = np.array( params ).reshape(self.param_dim, -1)
        batches = params.shape[1]
        # print('Sim:', params)
        for i in range(0, batches):
            x = params[0, i]
            y = params[1, i]
            # print(x, y)
            mic_pair = self.choose_rand_mic_pair()
            itd = self.get_itd(x, y, mic_pair)
            temp = []
            for _ in range(10):
                temp.append(t.rvs(df=3, scale=0.01, loc=itd))
            results.append( [ np.mean(temp), np.std(temp) ] )
        # print(results)
        # print('mean:', np.mean(results, axis=1))
        # print('std:', np.std(y_obs, axis=1))
        return results


    def get_itd(self, x, y, mic_pair):
        norm_1 = distance.euclidean((x, y), mic_pair[0])
        norm_2 = distance.euclidean((x, y), mic_pair[1])
        return np.abs( norm_1 - norm_2 )


    def choose_rand_mic_pair(self):
        n = random.random()
        if n < 0.5:
            return self.mic_pair_1
        else:
            return self.mic_pair_2


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        m = elfi.new_model()

        elfi.Prior('uniform', -2, 4, name='x')
        elfi.Prior('uniform', -2, 4, name='y')
        params = [m['x'], m['y']]

        if true_params is None:
            true_params = [[1.5, 1]]
        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        y_obs = [np.mean(y_obs), np.std(y_obs)]

        elfi.Simulator(self.func, *params, observed=y_obs, name='sim')
        # elfi.Summary(partial(np.mean, axis=1), m['sim'], name='Mean')
        # elfi.Summary(partial(np.std, axis=1), m['sim'], name='Std')
        elfi.Distance('euclidean', m['sim'], name='dist')

        return m