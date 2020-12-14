import numpy as np
import operator
import elfi
import elfi.examples.bdm_dgp_simulator.simulator as si
import elfi.examples.bdm_dgp_simulator.elfi_operations as ops

class bdm_simulator:
    def __init__(self):
        # Observation pediod in years
        self.t_obs = 2

        # Some bounds that discard unrealistic initial values to optimize the computation
        self.mean_obs_bounds = (0, 350)
        # Upper bounds for t1 and a1
        self.t1_bound = 30
        self.a1_bound = 40

        # Upper bound for the largest allowed cluster size within the observation period.
        # These are chosen to eliminate outcomes that are clearly different from the
        # observed data early
        self.cluster_size_bound = 80
        # Restrict warmup between 15 and 300 years
        self.warmup_bounds = (15, 300)

        # Set observed data and a fixed value for delta_2
        # self.y0 = ops.get_SF_data(self.cluster_size_bound)


    def func(self, *params, n_obs=100, batch_size=1, random_state=None):
        results = list()
        batches = len(params)

        # {'R1':(1.01, 12), 'R2': (0.01, 0.4),'burden': (120, 220), 't1':(0.01, 30)}
        for i in range(0, batches):

            R1 = params[i][0] # R2, t1
            R2 = params[i][1] # burden
            burden  = params[i][2] # R1, t1
            t1 = params[i][3] # R1, t1
            d1 = ops.Rt_to_d(R1, t1)
            d2 = 5.95
            a2 = R2 * d2
            a1 = ops.Rt_to_a(R1, t1)
            res = ops.simulator(burden, a2, d2, a1, d1, 2, self.cluster_size_bound, self.warmup_bounds)

            #print('')
            #print(self.y0)
            #print(self.y0[0])
            #print(res)
            #raise ValueError

            n_obs = res['n_obs']
            n_clusters = res['n_clusters']
            largest = res['largest']
            clusters = res['clusters']
            obs_times = res['obs_times']
            d = ops.distance(n_obs, n_clusters, largest, clusters, obs_times, self.y0_sum)
            # results.append([n_obs, n_clusters, largest, clusters, obs_times])
            results.append(d)
        # print(results)
        return results

    def identity(self, n_obs, n_clusters, largest, clusters, obs_times):
        return [n_obs, n_clusters, largest, clusters, obs_times]

    def get_model(self, n_obs=100, true_params=None, seed_obs=None):
        """Return a complete model in inference task.

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
        
        m = elfi.new_model()
        burden = elfi.Prior('normal', 200, 30, name='burden')

        joint = elfi.RandomVariable(ops.JointPrior, burden, self.mean_obs_bounds,
                                    self.t1_bound, self.a1_bound)

        # DummyPrior takes a marginal from the joint prior
        R2 = elfi.Prior(ops.DummyPrior, joint, 0, name='R2')
        R1 = elfi.Prior(ops.DummyPrior, joint, 1, name='R1')
        t1 = elfi.Prior(ops.DummyPrior, joint, 2, name='t1')

        # Turn the epidemiological parameters to rate parameters for the simulator
        d1 = elfi.Operation(ops.Rt_to_d, R1, t1)
        d2 = 5.95
        a2 = elfi.Operation(operator.mul, R2, d2)
        a1 = elfi.Operation(ops.Rt_to_a, R1, t1)

        if true_params is None:
            y0_burden = 192
            y0_R2 = 0.09
            y0_R1 = 5.88
            y0_t1 = 6.74

            y0_d1 = ops.Rt_to_d(y0_R1, y0_t1)
            y0_a2 = operator.mul(y0_R2, d2)
            y0_a1 = ops.Rt_to_a(y0_R1, y0_t1)
            self.y0 = ops.simulator(y0_burden, y0_a2, d2, y0_a1, y0_d1, 2,
                                    self.cluster_size_bound, self.warmup_bounds)

            self.y0_sum = [self.y0['n_obs'], self.y0['n_clusters'], self.y0['largest'], self.y0['clusters'], self.y0['obs_times']]

        # Add the simulator
        sim = elfi.Simulator(ops.simulator, burden, a2, d2, a1, d1, 2,
                             self.cluster_size_bound, self.warmup_bounds, observed=self.y0)

        # Summaries extracted from the simulator output
        n_obs = elfi.Summary(ops.pick, sim, 'n_obs')
        n_clusters = elfi.Summary(ops.pick, sim, 'n_clusters')
        largest = elfi.Summary(ops.pick, sim, 'largest')
        clusters = elfi.Summary(ops.pick, sim, 'clusters')
        obs_times = elfi.Summary(ops.pick, sim, 'obs_times')

        sim = elfi.Operation(ops.distance, n_obs, n_clusters, largest, clusters,
                                obs_times, self.y0_sum, name = 'sim')

        # Distance
        dist = elfi.Discrepancy(ops.distance, n_obs, n_clusters, largest, clusters,
                                obs_times, name = 'dist')
        return m
