import elfi
import numpy as np
import math

class cosmological_inflation:

    def __init__(self) -> None:
        self.param_dim = 5


    def func(self, *params, n_obs=100, batch_size=1, random_state=None):
        results = list()
        params = np.array( params ).reshape(self.param_dim, -1)
        batches = params.shape[1]
        # print('Sim:', params)

        for i in range(0, batches):
            ns = np.float64(params[0, i])
            kc = np.float64(params[1, i])
            alpha = np.float64(params[2, i])
            r_star = np.float64(params[3, i])
            As = np.float64(params[4, i])

            k = np.random.uniform(0.0008, 0.00085)
            y = k / kc
            temp = self.get_power_law_spectrum(As, y, alpha, k, ns, r_star) 
            if temp == float("-inf") or temp < -1e7:
                temp = -1e7
            elif temp == float("inf") or temp > 1e7:
                temp = 1e7
            elif math.isnan(temp):  
                temp = 1e7          

            results.append(temp)
        return results


    def get_power_law_spectrum(self, As, y, alpha, k, ns, r_star):
        return As * (1 - np.exp((0.75*y)**alpha) ) * k**(ns-1) * self.get_T2(y, r_star)


    def get_T2(self, y, r_star):
        term_1 = 1-3*(r_star-1) * (1./y) * ( (1 - (1./y**2)) * np.sin(2 * y) + (2. / y) * np.cos(2*y) ) 
        term_2 = 4.5 * r_star**2 * ( 1./y**2 ) * (1 + (1./ y**2)) * (1 + 1./y**2 + (1 - 1./y**2) * np.cos(2*y) - 2./y * np.sin(2*y) )
        return term_1 + term_2


    def get_model(self, n_obs=10, true_params=None, seed_obs=None):
        m = elfi.new_model()

        # Parameters: ns, kc, alpha, r_star, As
        # priors from Sinha and Souradeep 2006.
        elfi.Prior('uniform', 0.5, 1.0, name='ns')
        elfi.Prior('uniform', 1e-7, 1e-3, name='kc')
        elfi.Prior('uniform', 0., 10., name='alpha')
        elfi.Prior('uniform', 0., 1., name='r_star')
        elfi.Prior('uniform', 2.7, 1.3, name='As')
        params = [m['ns'], m['kc'], m['alpha'], m['r_star'], m['As']]

        if true_params is None:
            true_params = [[0.96, 0.0003, 0.58, 0.75, 3.35 ]]
        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))

        elfi.Simulator(self.func, *params, observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m