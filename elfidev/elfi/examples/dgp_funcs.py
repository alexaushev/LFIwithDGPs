import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import elfi
from matplotlib.ticker import StrMethodFormatter
from sklearn.neighbors import KernelDensity

# this function used most of the code from: https://github.com/hughsalimbeni/DGPs_with_IWVI
def plot_posterior_samples(target_model, x_counts = 1000, samples = 100, points = True, kde = True):
    m = target_model
    bounds = m.bounds
    S = samples
    
    if type(m).__name__ == 'DGPRegression':
        posterior = m.get_posterior
    elif type(m).__name__ == 'GPyRegression':
        posterior = m.get_posterior
    else:
        raise ValueError("The target_model should be either 'DGPRegression'"
                         "or 'GpyRegression'")
    
    Xs = np.linspace(*bounds[0], x_counts)
    samples = posterior(Xs, size=S)
    samples = samples[:, :, 0]
    ydif = (max(m.Y) - min(m.Y)) * 0.15
    levels = np.linspace(min(m.Y) - ydif, max(m.Y) + ydif, 1000)

    ax = plt.gca()
    # ax.set_ylim(min(levels), max(levels))
    # ax.set_ylim(min(levels), 1.0)
    # ax.set_xlim(min(Xs), max(Xs))
    plt.xticks(np.arange(0, 100, step = 10))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$d(x_\theta, x_{obs})$")

    if kde == True:
        cs = np.zeros((len(Xs), len(levels)))
        for i, Ss in enumerate(samples.T):
            bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
            kde = KernelDensity(bandwidth=float(bandwidth))

            kde.fit(Ss.reshape(-1, 1))
            for j, level in enumerate(levels):
                cs[i, j] = kde.score(np.array(level).reshape(1, 1))
        ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T), cmap='Blues_r') # , alpha=0.1)

    if points == True:
        ax.scatter(m.X, m.Y, s = 15, color = "red", zorder=10)
    return


def plot_variance(target_model, x_counts = 1000, samples = 100, points = True, kde = True):
    m = target_model
    bounds = m.bounds
    S = samples
    
    if type(m).__name__ == 'DGPRegression':
        posterior = m.get_posterior
    elif type(m).__name__ == 'GPyRegression':
        posterior = m.get_posterior
    else:
        raise ValueError("The target_model should be either 'DGPRegression'"
                         "or 'GpyRegression'")
    
    Xs = np.linspace(*bounds[0], x_counts)
    samples, res_mean, res_var = posterior(Xs, size=S)
    samples = samples[:, :, 0]
    ydif = (max(m.Y) - min(m.Y)) * 0.15
    levels = np.linspace(min(m.Y) - ydif, max(m.Y) + ydif, 1000)

    ax = plt.gca()
    # ax.set_ylim(min(levels), max(levels))
    # ax.set_ylim(min(levels), 1.0)
    # ax.set_xlim(min(Xs), max(Xs))
    plt.xticks(np.arange(0, 100, step = 10))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$d(x_\theta, x_{obs})$")

    # build variance
    #print(res_mean)
    #print(res_var)
    print(len(res_var))
    clrs = ["red", "green", "blue"]
    for i in range(0, len(res_var)):
        cov = res_var[i]

        if len(cov[-1].flatten()) != len(Xs):
            print(cov[-1].flatten()[0])
            continue

        for j in range(0, len(cov)):
            ax.scatter(Xs.flatten(), cov[j].flatten(), s = 1, color = clrs[i], zorder=10)
    return


def plot_d(func, true_params, bounds):
    x = np.linspace(*bounds['t1'], num = 4000)
    y_true = func(*true_params)
    ys = func(x)
    d = (ys - y_true)**2
    sqrtd = np.sqrt(d)

    fig = plt.gcf()
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.scatter(x, sqrtd, color='black', s=1)
    # plt.yticks([])
    plt.rcParams.update({'font.size': 10})
    plt.xticks(np.arange(0, 101, step = 50))
    
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"")

    fig.set_size_inches(2.5,2.1)
    plt.savefig('plots/func.png', dpi=600, bbox_inches = 'tight')
    plt.close()
        
    # plt.xticks(np.arange(min(x), max(x)+1, 10.0))
    # plt.axvline(0, color = 'black')
    # plt.axhline(0, color = 'black')
    return    


def plot_func(func, bounds):
    x = np.linspace(*bounds[0], num = 2000)
    ys = func(x)

    plt.scatter(x, ys, color='black')
    plt.xticks(np.arange(min(x), max(x)+1, 10.0))
    plt.axvline(0, color = 'black')
    plt.axhline(0, color = 'black')
    return    


class xcosx:
    def __init__(self):
        pass
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        # Make inputs 2d arrays for broadcasting with w
        t1 = np.asanyarray(t1).reshape((-1, 1))
        random_state = random_state or np.random

        # i.i.d. sequence ~ N(0,1)
        w = random_state.randn(batch_size, 1)
        x = t1 * np.cos(t1 * np.pi / 180)
        return x

    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [90]
        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 360, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m


class logistic:
    def __init__(self):
        pass
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        t1 = np.asanyarray(t1).reshape((-1, 1))
        random_state = random_state or np.random
        w = random_state.randn(batch_size, 1)
        x = 1 / (1 + np.exp(-0.1 * (t1 - 180)))
        return x


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [90]

        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 360, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m


class multigaussian:
    def __init__(self):
        pass
    
    def N(self, mu, sigma, x):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1/2)*np.power((x - mu)/sigma, 2))
        
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        t1 = np.asanyarray(t1).reshape((-1, 1))
        random_state = random_state or np.random
        w = random_state.randn(batch_size, 1) / 400
        x = self.N(30, 15, t1) + self.N(60, 5, t1) + self.N(100, 4, t1) + w
        return x


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [50]

        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 100, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m


# UNUSUAL UNCERTAINTIES ==== ====

class multimodal_logistic:
    def __init__(self, n=0.5, offset=50, noise=0.01):
        self.offset = offset
        self.n = n
        self.noise=noise
        
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        t1 = np.asanyarray(t1).reshape((-1, 1))
        random_state = random_state or np.random

        batch_size = len(t1)
        w = random_state.randn(batch_size, 1)
        source = random_state.uniform(size=batch_size)
        x = list()
        # print(source)
        #print(w)
        
        for el in range(batch_size):
            if source[el] < self.n:
                temp = 1 / (1 + np.exp(-0.1 * (t1[el] - self.offset))) + w[el] * self.noise
            else:
                temp = -1 / (1 + np.exp(-0.1 * (t1[el] - self.offset))) + w[el] * self.noise + 1
            x.append(temp)
        #zprint(x)
        return np.array(x)


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [20]

        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 100, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m



class beta_x:
    def __init__(self):
        pass
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        t1 = np.asanyarray(t1).reshape((-1, 1))
        random_state = random_state or np.random
        batch_size = len(t1)
        x = list()

        #print(t1)
        for el in range(batch_size):
            w = np.random.beta(t1[el] + 1, 5, size = 1) + np.random.beta(5, t1[el] + 1, size = 1)
            x.append(w)
        # print(x)
        return np.array(x)


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [20]

        y_obs = self.func(*true_params, batch_size=100, random_state=np.random.RandomState(seed_obs))
        y_obs = np.mean(y_obs)
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 100, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m




class bigaussian:
    def __init__(self):
        pass
    
    def N(self, mu, sigma, x):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1/2)*np.power((x - mu)/sigma, 2))
        
    
    def func(self, t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        t1 = np.asanyarray(t1).reshape((-1, 1))
        batch_size = len(t1)
        random_state = random_state or np.random
        w = random_state.randn(batch_size, 1) / 10000
        source = random_state.uniform(size=batch_size)

        x = []
        for el in range(batch_size):
            if source[el] < 0.4:
                k1, k2 = 1, 0
            elif source[el] >= 0.4:
                k1, k2 = 0, 1
        
            temp = 100* (self.N(0, 50, t1[el]) * k1 + self.N(60, 55, t1[el]) * k2 +  w[el])
            x.append(temp)
        return np.array(x).flatten()


    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete MA2 model in inference task.

        Parameters
        ----------
        n_obs : int, optional
            observation length of the MA2 process
        true_params : list, optional
            parameters with which the observed data is generated
        seed_obs : int, optional
            seed for the observed data generation

        Returns
        -------
        m : elfi.ElfiModel

        """
        if true_params is None:
            true_params = [60]

        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        m = elfi.ElfiModel()
        elfi.Prior(ss.uniform, 0, 100, model=m, name='t1') 
        elfi.Simulator(self.func, m['t1'], observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        return m
