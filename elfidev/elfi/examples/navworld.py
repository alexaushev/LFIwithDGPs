import numpy as np
import elfi

from simple_rl.run_experiments import run_single_agent_on_mdp
from simple_rl.tasks import NavigationWorldMDP
from simple_rl.tasks.navigation.NavigationStateClass import NavigationWorldState
from simple_rl.agents import QLearningAgent



def plot_parameters(pars, md):
    cur_cell_rewards = [pars["white"][0], pars["yellow"][0], pars["red"][0],
                        pars["green"][0], pars["purple"][0], -500]
    # cur_cell_rewards = pars
    print(cur_cell_rewards)
    md.mdp = NavigationWorldMDP(width=md.side, height=md.side,
                                nav_cell_types=md.nav_cell_types,
                                nav_cell_rewards=cur_cell_rewards,
                                nav_cell_p_or_locs=md.nav_cell_p_or_locs,
                                goal_cell_types=md.goal_cell_types,
                                goal_cell_rewards=md.goal_rew,
                                goal_cell_locs=md.goal_cell_loc,
                                init_loc=md.start_loc,
                                rand_init=False, gamma=0.95, slip_prob = 0, step_cost=0)

    md.agent = QLearningAgent(md.mdp.get_actions(), epsilon=md.eps)
    run_single_agent_on_mdp(md.agent, md.mdp, episodes=md.episodes, steps=md.steps)
    md.agent.epsilon = 0
    md.mdp.slip_prob = 0
    _, steps_taken, reward, states = md.run_experiment(md.agent, md.mdp)
    # print('Best result observation:')
    print([md.count_turns(states), steps_taken, reward])
    # print('Observed data result:')
    # print(md.observed_data)
    md.mdp.visualize_grid(trajectories=[states], plot=False)
    return [md.count_turns(states), steps_taken, reward]

class navworld_simulator:
    def __init__(self, ep = 8000, steps = 100, side = 15, slip = 0.2, eps = 0.5,
                 start = (2, 2), end = [[(14, 14)]], goal_rew = [100]):
        # number of episodes per mdp and agent
        self.episodes = ep
        self.steps = steps
        self.side = side
        self.nav_cell_types = ["white", "yellow", "red", "green", "purple", "black"]
        self.goal_cell_types = ["blue"]
        self.slip = slip
        self.start_loc = start
        self.goal_cell_loc = end
        self.goal_rew = goal_rew
        self.prev_cell_rewards = None
        self.eps = eps
        
        # generate map for the rest of experiments
        self.mdp = NavigationWorldMDP(width=self.side, height=self.side,
                                      nav_cell_types = self.nav_cell_types,
                                      nav_cell_rewards = [0.0, -1.0, -1.0, -5.0, -10.0, -500],
                                      nav_cell_p_or_locs=[0.1, 0.2, 0.2, 0.25, 0.25,
                                                          [(13, 12),(13, 13),  (15, 13)]],
                                      goal_cell_types = self.goal_cell_types,
                                      goal_cell_rewards=self.goal_rew,
                                      goal_cell_locs=self.goal_cell_loc,
                                      init_loc=self.start_loc, rand_init=False,
                                      gamma=0.95, slip_prob=0, step_cost=0)

        # self.agent = QLearningAgent(self.mdp.get_actions(), epsilon=0.1)

        # get observed data
        self.agent = QLearningAgent(self.mdp.get_actions(), epsilon=self.eps)
        #run_single_agent_on_mdp(self.agent, self.mdp, episodes=self.episodes, steps=self.steps)
        #self.agent.epsilon = 0
        #self.mdp.slip_prob = 0 
        #_, steps_taken, reward, states = self.run_experiment(self.agent, self.mdp)
        self.observed_data = [9, 24, 51]
        
        # save grid for future simulations
        self.grid = self.get_grid_locs(self.mdp)
        self.nav_cell_p_or_locs = []
        self.nav_cell_types = self.nav_cell_types[:-1]
        for color in self.nav_cell_types:
            if color == 'white':
                self.nav_cell_p_or_locs.append(1.0)
            else:
                self.nav_cell_p_or_locs.append(self.grid[color])

        print(self.observed_data)
        # self.mdp.visualize_grid(trajectories=[states], traj_colors_auto=False)
        pass


    def count_turns(self, states):       
        prev_direction = None
        prev_state = states[0]
        turns = 0
        
        for i in range(1, len(states)):
            direction = "x" if abs(prev_state.x - states[i].x) != 0 else "y"
            if prev_direction != direction and prev_direction != None:
                turns += 1
                
            prev_direction = direction
            prev_state = states[i]
        return turns    


    def get_observed_data(self):
        return self.observed_data


    def get_grid_locs(self, mdp):
        cell_types = mdp.nav_cell_types
        cell_types.append("blue")
        grid_locs = dict.fromkeys(cell_types)
        for keys in grid_locs.keys():
            grid_locs[keys] = []
        
        for x in range(1, self.side + 1):
            for y in range(1, self.side + 1):
                r, c = mdp._xy_to_rowcol(x, y)
                grid_locs[mdp.nav_cell_types[mdp.map_state_cell_id[r, c]]].append((x, y))
        return grid_locs


    def get_q_func(self, agent):
        q_func = dict()
        for state, actiond in agent.q_func.items():
            q_func[state] = dict()
            for action, q_val in actiond.items():
                q_func[state][action] = q_val
                    
        return q_func


    def run_experiment(self, agent, mdp, steps = 100):
        states = [mdp.get_init_state()]
        total_reward = 0
        state = mdp.get_init_state()
        gamma = mdp.get_gamma()
        reward = 0
        
        for step in range(1, steps + 1):
            action = agent.act(state, reward, False)

            reward, next_state = mdp.execute_agent_action(action)
            states.append(next_state)
            total_reward += reward

            if next_state.is_terminal():
                mdp.reset()
                return True, step, total_reward, states
            state = next_state

        mdp.reset()
        agent.end_of_episode()
        return False, steps, total_reward, states
    

    def func(self, *params, n_obs=100, batch_size=1, random_state=None):
        """Generate a sequence of samples from the Open AI env.

        Parameters
        ----------
        params : array of envs
        random_state : RandomState, optional

        """
        
        # fix locations instead of probabilities! fixed map multiple init_locs!
        rewards = []

        if isinstance(params, tuple):
            framework = 'elfi'
        else:
            framework = 'delfi'

        if framework == 'delfi':
            batch_size = len(params)
        else:
            batch_size = len(params[0])
        # print()
        
        for i in range(batch_size):
            
            # cur_cell_rewards = [x[i] for x in params]
            # print(i)
            # print(params)
            if framework == 'delfi':
                cur_cell_rewards = list(params[i])
            else:
                cur_cell_rewards = [x[i] for x in params]

            # reward for black cells is fixed
            cur_cell_rewards.append(-500)

            if self.prev_cell_rewards != cur_cell_rewards:
                self.mdp = NavigationWorldMDP(width=self.side, height=self.side,
                                              nav_cell_types=self.nav_cell_types,
                                              nav_cell_rewards=cur_cell_rewards,
                                              nav_cell_p_or_locs=self.nav_cell_p_or_locs,
                                              goal_cell_types = self.goal_cell_types,
                                              goal_cell_rewards=self.goal_rew,
                                              goal_cell_locs=self.goal_cell_loc,
                                              init_loc=self.start_loc,
                                              rand_init=False,
                                              slip_prob = 0)

                self.agent = QLearningAgent(self.mdp.get_actions(), epsilon=self.eps)
                run_single_agent_on_mdp(self.agent, self.mdp, episodes=self.episodes, steps=self.steps)
                

            self.agent.epsilon = 0
            self.mdp.slip_prob = self.slip

            # print('Parameters:')
            # print(cur_cell_rewards)
            for j in range(1):
                finished, steps_taken, reward, states = self.run_experiment(self.agent, self.mdp)
                turns = self.count_turns(states)
                ep_reward = [turns, steps_taken, reward]
                # print('Corresponding reward:')
                # print([turns, steps_taken, reward])
                # self.mdp.visualize_grid(trajectories=[states], traj_colors_auto=False)
                
            rewards.append(ep_reward)
            
            self.prev_cell_rewards = cur_cell_rewards
        return rewards

    def discrepancyI(s1, s2, obs):
        traj_obs = set(obs[0][0])
        rew_obs = obs[0][1]
        dis = list()

        for entry in s2:
            traj = set(entry[0])
            rew = entry[1]
            traj_dis = max(len(traj_obs - traj), len(traj - traj_obs)) 
            rew_dis = np.abs(rew_obs - rew)
            dis.append(traj_dis + rew_dis)

        dis = np.array(dis)
        print('Discrepancy:')
        print(dis)
        return dis


    def discrepancyII(s1, s2, obs):
        ws = [6, 3, 1]
        obs = obs[0]
        dis = list()

        #print('+')
        #rint(obs)

        for entry in s2:
            #print(entry)
            #print(ws)
            #print(np.multiply(entry, ws))
            #print(np.multiply(obs, ws))
            dis.append( np.linalg.norm(np.multiply(entry, ws) - np.multiply(obs, ws)) )

        dis = np.array(dis)
        print('Discrepancy:')
        print(dis)
        return dis
        

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
        if true_params is None:
            elfi.Prior('uniform', -20.0, 20.0, model=m, name='white')
            elfi.Prior('uniform', -20.0, 20.0, model=m, name='yellow')
            elfi.Prior('uniform', -20.0, 20.0, model=m, name='red')
            elfi.Prior('uniform', -20.0, 20.0, model=m, name='green')
            elfi.Prior('uniform', -20.0, 20.0, model=m, name='purple')
            params = [m['white'], m['yellow'], m['red'],
                      m['green'], m['purple']]
            
        y_obs = self.get_observed_data()
        elfi.Simulator(self.func, *params, observed=y_obs, name='sim')
        elfi.Distance('euclidean', m['sim'], name='dist')
        # elfi.Distance(self.discrepancyII, m['DGP'], name='d')
        return m

