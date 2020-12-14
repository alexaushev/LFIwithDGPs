"""This module contains functions for interactive ("iterative") plotting."""

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from sklearn.neighbors import KernelDensity


def plot_sample(samples, nodes=None, n=-1, displays=None, **options):
    """Plot a scatterplot of samples.

    Experimental, only dims 1-2 supported.

    Parameters
    ----------
    samples : Sample
    nodes : str or list[str], optional
    n : int, optional
        Number of plotted samples [0, n).
    displays : IPython.display.HTML

    """
    axes = _prepare_axes(options)

    nodes = nodes or sorted(samples.keys())[:2]
    if isinstance(nodes, str):
        nodes = [nodes]

    if len(nodes) == 1:
        axes.set_xlabel(nodes[0])
        axes.hist(samples[nodes[0]][:n])
    else:
        if len(nodes) > 2:
            logger.warning('Over 2-dimensional plots not supported. Falling back to 2d'
                           'projection.')
        axes.set_xlabel(nodes[0])
        axes.set_ylabel(nodes[1])
        axes.scatter(samples[nodes[0]][:n], samples[nodes[1]][:n])

    _update_interactive(displays, options)

    if options.get('close'):
        plt.close()


def get_axes(**options):
    """Get an Axes object from `options`, or create one if needed."""
    if 'axes' in options:
        return options['axes']
    return plt.gca()


def _update_interactive(displays, options):
    displays = displays or []
    if options.get('interactive'):
        from IPython import display
        display.clear_output(wait=True)
        displays.insert(0, plt.gcf())
        display.display(*displays)


def _prepare_axes(options):
    axes = get_axes(**options)
    ion = options.get('interactive')

    if ion:
        axes.clear()

    if options.get('xlim'):
        axes.set_xlim(options.get('xlim'))
    if options.get('ylim'):
        axes.set_ylim(options.get('ylim'))

    return axes


def plot_func(gp, acq_fn, nodes=None, points=None, new_points=None, title=None, **options):
    """Plot a contour of a function.

    Experimental, only 2D supported.

    Parameters
    ----------
    fn : callable
    bounds : list[arraylike]
        Bounds for the plot, e.g. [(0, 1), (0,1)].
    nodes : list[str], optional
    points : arraylike, optional
        Additional points to plot.
    title : str, optional

    """
    #print("Plot")
    mean_fn = gp.predict_mean
    var_fn = gp.predict_var
    bounds = gp.bounds
    
    ax = get_axes(**options)

    x = np.linspace(*bounds[0], num = 1000)

    def flatten(l):
        return [item for sublist in l for item in sublist]

    mean_vals = np.array(mean_fn(x)).flatten()
    var_vals = np.array(var_fn(x)).flatten()
    
    upper_bound = mean_vals + var_vals ** 0.5
    lower_bound = mean_vals - var_vals ** 0.5

    
        #min_acq = [min_acq] * len(x)
        #print(len(min_acq))
        #print(len(acq_vals))

    if ax:
        plt.sca(ax)
    plt.cla()

    # plt.axhline(0, color = 'black')
    if title:
        plt.title(title)
        
    #plt.plot(x, mean_vals, color='black')
    # plt.plot(x, lowe, color='grey', linestyle='dashed')
    # plt.plot(x, mean_vals + var_vals ** 0.5, color='grey', linestyle='dashed')
    plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.2)
    if acq_fn is not None:
        print('interactive.py: Acqusition!')
        acq_vals = np.array(acq_fn(x)).flatten()
        #acq_mean = np.mean(acq_vals, 0)
        #acq_std = np.std(acq_vals, 0)
        #acq_vals = (acq_vals - acq_mean) / acq_std
        
        min_acq = np.amin(acq_vals) + min(gp.Y)
        plt.plot(x, acq_vals, color='green')
        plt.fill_between(x, acq_vals, min_acq, color='green', alpha=0.2)


    #print(x.shape)
    #print(mean_vals.shape)
    #print(upper_bound.shape)
    
    if points is not None:
        #print(len(points))
        #print(len(gp.Y.value))
        plt.scatter(points, gp.Y, s = 15, color = "red", zorder=10)
    ##if new_points is not None:
     #   plt.scatter(new_points, gp.Y, s = 15, color = "red")
    
        #if options.get('interactive'):
        #    plt.scatter(points[-1, 0], points[-1, 1], color='r')
            
    #plt.axvline(0, color = 'black')
    plt.xlim(bounds[0])
    
    plt.xticks(np.arange(min(x), max(x)+1, 10.0))

    if nodes:
        plt.xlabel(nodes[0])


def plot_posterior(gp, **options):
    bounds = gp.bounds
    posterior = gp.get_posterior
    S = gp.num_posterior_samples
    
    ax = get_axes(**options)
    Xs = np.linspace(*bounds[0], num = 1000)
    samples = posterior(Xs, S)[:, :, 0]
    # print(samples)
    ydif = (max(gp.Y) - min(gp.Y)) * 0.15
    levels = np.linspace(min(gp.Y) - ydif, max(gp.Y) + ydif, 1000)

    ax.set_ylim(min(levels), max(levels))
    ax.set_xlim(min(Xs), max(Xs))

    cs = np.zeros((len(Xs), len(levels)))
    for i, Ss in enumerate(samples.T):
        bandwidth = 1.06 * np.std(Ss) * len(Ss) ** (-1. / 5)  # Silverman's (1986) rule of thumb.
        kde = KernelDensity(bandwidth=float(bandwidth))

        kde.fit(Ss.reshape(-1, 1))
        for j, level in enumerate(levels):
            cs[i, j] = kde.score(np.array(level).reshape(1, 1))
    ax.pcolormesh(Xs.flatten(), levels, np.exp(cs.T), cmap='Blues_r') # , alpha=0.1)
    ax.scatter(gp.X, gp.Y, s = 15, color = "red", zorder=10)
    

    '''for j in range(0, 5):
        samples = posterior(x, S)
        for sample in samples:
            vals = np.array(sample).flatten()
            plt.scatter(x, vals, color='blue')

    plt.scatter(gp.X, gp.Y, s = 15, color = "red")'''
    #plt.show()


def draw_contour(fn, bounds, nodes=None, points=None, title=None, **options):
    """Plot a contour of a function.

    Experimental, only 2D supported.

    Parameters
    ----------
    fn : callable
    bounds : list[arraylike]
        Bounds for the plot, e.g. [(0, 1), (0,1)].
    nodes : list[str], optional
    points : arraylike, optional
        Additional points to plot.
    title : str, optional

    """
    ax = get_axes(**options)

    x, y = np.meshgrid(np.linspace(*bounds[0]), np.linspace(*bounds[1]))
    z = fn(np.c_[x.reshape(-1), y.reshape(-1)])

    if ax:
        plt.sca(ax)
    plt.cla()

    if title:
        plt.title(title)
    try:
        plt.contour(x, y, z.reshape(x.shape))
    except ValueError:
        logger.warning('Could not draw a contour plot')
    if points is not None:
        plt.scatter(points[:-1, 0], points[:-1, 1])
        if options.get('interactive'):
            plt.scatter(points[-1, 0], points[-1, 1], color='r')

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])

    if nodes:
        plt.xlabel(nodes[0])
        plt.ylabel(nodes[1])
