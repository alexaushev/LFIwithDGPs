import numpy as np
import elfi

import tensorflow as tf
import gym
from baselines import deepq

class control_env:
    def __init__(self, env_name):
        self.env_name = env_name
        if self.env_name == "CartPole-v0":
            self.load_path = "cartpole_model.pkl"
        elif self.env_name == "MountainCar-v0":
            self.load_path = "mountaincar_model.pkl"
        self.sess = tf.Session().__enter__()
        self.env = gym.make(self.env_name)
        self.act = deepq.learn(self.env, network='mlp', total_timesteps=0, load_path=self.load_path)
        pass


    def func(self, *params, n_obs=100, batch_size=1, random_state=None):
        """Generate a sequence of samples from the Open AI env.

        Parameters
        ----------
        params : array of envs
        random_state : RandomState, optional

        """
        batch_size = len(params[0])
        rewards = []
        # print('unbatched params:')
        # print(params)
        for i in range(batch_size):
            cur_pars = [x[i] for x in params]
            #print('Params:')
            # print(cur_pars)
            self.env = self.apply_pars(self.env, cur_pars)
            obs, done = self.env.reset(), False
            episode_rew = 0
            while not done:
                # self.env.render()
                obs, rew, done, _ = self.env.step(self.act(obs[None])[0])
                episode_rew += rew
            
            # print('Reward:')
            print(episode_rew)
            rewards.append(episode_rew)
        return rewards


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
        
        m = elfi.ElfiModel()
        if true_params is None and self.env_name == "CartPole-v0":
            # gravity, masscart, masspole, length, force_mag
            true_params = [[9.8], [1.0], [0.1], [0.5], [10.0]]
            elfi.Prior('uniform', 5.0, 10.0, model=m, name='gravity')
            elfi.Prior('uniform', 0.1, 5, model=m, name='masscart')
            elfi.Prior('uniform', 0.1, 5, model=m, name='masspole')
            elfi.Prior('uniform', 0.1, 5, model=m, name='length')
            elfi.Prior('uniform', 5.0, 10.0, model=m, name='force_mag')
            params = [m['gravity'], m['masscart'], m['masspole'],
                      m['length'], m['force_mag']]
            
        elif true_params is None and self.env_name == "MountainCar-v0":
            # goal_position, goal_velocity, force, gravity
            true_params = [0.5, 0, 0.001, 0.0025]
            elfi.Prior('uniform', -1.2, 0.6, model=m, name='goal_position')
            elfi.Prior('uniform', 0, 5.0, model=m, name='goal_velocity')
            elfi.Prior('uniform', 0.0001, 0.01, model=m, name='force')
            elfi.Prior('uniform', 0.0001, 0.01, model=m, name='gravity')
            params = [m['goal_position'], m['goal_velocity'],
                      m['force'], m['gravity']]

        y_obs = self.func(*true_params, random_state=np.random.RandomState(seed_obs))
        elfi.Simulator(self.func, *params, observed=y_obs, name='DGP')
        elfi.Distance('euclidean', m['DGP'], name='d')
        return m


    def apply_pars(self, env, params):
        if self.env_name == "CartPole-v0":
            env.gravity = params[0]
            env.masscart = params[1]
            env.masspole = params[2]
            env.total_mass = (env.masspole + env.masscart)
            env.length = params[3] # actually half the pole's length
            env.polemass_length = (env.masspole * env.length)
            env.force_mag = params[4]
        elif self.env_name == "MountainCar-v0":
            env.goal_position = params[0]
            env.goal_velocity = params[1]
            env.force=params[2]
            env.gravity=params[3]   
        return env
