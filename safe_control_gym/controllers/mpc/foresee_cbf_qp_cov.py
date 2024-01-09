"""Model Predictive Control with a Gaussian Process model.

Based on:
    * L. Hewing, J. Kabzan and M. N. Zeilinger, "Cautious Model Predictive Control Using Gaussian Process Regression,"
     in IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2736-2743, Nov. 2020, doi: 10.1109/TCST.2019.2949757.

Implementation details:
    1. The previous time step MPC solution is used to compute the set constraints and GP dynamics rollout.
       Here, the dynamics are rolled out using the Mean Equivelence method, the fastest, but least accurate.
    2. The GP is approximated using the Fully Independent Training Conditional (FITC) outlined in
        * J. Quinonero-Candela, C. E. Rasmussen, and R. Herbrich, “A unifying view of sparse approximate Gaussian process regression,”
          Journal of Machine Learning Research, vol. 6, pp. 1935–1959, 2005.
          https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
        * E. Snelson and Z. Ghahramani, “Sparse gaussian processes using pseudo-inputs,” in Advances in Neural Information Processing
          Systems, Y. Weiss, B. Scholkopf, and J. C. Platt, Eds., 2006, pp. 1257–1264.
       and the inducing points are the previous MPC solution.
    3. Each dimension of the learned error dynamics is an independent Zero Mean SE Kernel GP.

"""
import scipy
import numpy as np
import casadi as cs
import time
import torch
import gpytorch

import jax
import jax.numpy as jnp
from jax import jit, lax, grad, value_and_grad

from copy import deepcopy
from skopt.sampler import Lhs
from functools import partial
from sklearn.model_selection import train_test_split

from safe_control_gym.controllers.mpc.linear_mpc import LinearMPC, MPC
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import GaussianProcessCollection, ZeroMeanIndependentGPModel, covSEard
from safe_control_gym.envs.benchmark_env import Task

# New
from safe_control_gym.controllers.mpc.foresee_utils_jax import *
import pdb

class FORESEE_CBF_QP_COV(MPC):
    """MPC with Gaussian Process as dynamics residual and FORESEE for uncertainty propagation. 

    """
    predict = []
    predict_grad = []
    reward_func = []
    sigma_point_expand = []

    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            additional_constraints: list = None,
            use_prev_start: bool = True,
            train_iterations: int = 800,
            validation_iterations: int = 200,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            sparse_gp: bool = False,
            online_learning: bool = False,
            inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            output_dir: str = "results/temp",
            **kwargs
            ):
        """Initialize GP-MPC.

        Args:
            env_func (gym.Env): functionalized initialization of the environment.
            seed (int): random seed.
            horizon (int): MPC planning horizon.
            Q, R (np.array): cost weight matrix.
            use_prev_start (bool): Warmstart mpc with the previous solution.
            train_iterations (int): the number of training examples to use for each dimension of the GP.
            validation_iterations (int): the number of points to use use for the test set during training.
            optimization_iterations (list): the number of optimization iterations for each dimension of the GP.
            learning_rate (list): the learning rate for training each dimension of the GP.
            normalize_training_data (bool): Normalize the training data.
            use_gpu (bool): use GPU while training the gp.
            gp_model_path (str): path to a pretrained GP model. If None, will train a new one.
            output_dir (str): directory to store model and results.
            prob (float): desired probabilistic safety level.
            initial_rollout_std (float): the initial std (across all states) for the mean_eq rollout.
            inertial_prop (list): to initialize the inertial properties of the prior model.
            prior_param_coeff (float): constant multiplying factor to adjust the prior model intertial properties.
            input_mask (list): list of which input dimensions to use in GP model. If None, all are used.
            target_mask (list): list of which output dimensions to use in the GP model. If None, all are used.
            gp_approx (str): 'mean_eq' used mean equivalence rollout for the GP dynamics. Only one that works currently.
            online_learning (bool): if true, GP kernel values will be updated using past trajectory values.
            additional_constraints (list): list of Constraint objects defining additional constraints to be used.

        """
        self.prior_env_func = partial(env_func,
                                      inertial_prop=np.array(inertial_prop)*prior_param_coeff)
        self.prior_param_coeff = prior_param_coeff
        # Initialize the method using linear MPC.
        self.prior_ctrl = LinearMPC(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            use_prev_start=use_prev_start,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
        )
        self.prior_ctrl.reset()
        super().__init__(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            use_prev_start=use_prev_start,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs)
        
        # Setup controller parameters
        kx = 0.09 #0.06 #0.09 #0.2 #0.1 #0.5 # Attarction
        kv = 0.05 #0.1 #0.05 #0.20000002#  0.3 # Attraction
        krx = 0.4 #0.4 #0.8 # Repulsion
        kR = 60.0
        kRv = 10.0
        # krv = 1.0 # Repulsion
        # kT1x = -0.2
        # kT1y = 0.2
        # kT10 = 5.0
        # kT2x = 0.2
        # kT2y = 1.0
        # kT20 = 5.0
     
        self.adapt = True
        self.num_adapt_iterations = 10

        self.params = np.array([ kx, kv, krx, kR, kRv ])


        # Setup environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False)
        self.env_training = env_func(randomized_init=True)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.validation_iterations = validation_iterations
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.normalize_training_data = normalize_training_data
        self.use_gpu = use_gpu
        self.seed = seed
        self.prob = prob
        self.sparse_gp = sparse_gp
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]

        # print(f"Bd: {self.Bd}")
        # exit()
        self.gp_approx = gp_approx
        self.online_learning = online_learning
        self.last_obs = None
        self.last_action = None
        self.initial_rollout_std = initial_rollout_std

        

    def setup_prior_dynamics(self):
        """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics.

        """
        # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
        A, B = discretize_linear_system(self.prior_ctrl.dfdx, self.prior_ctrl.dfdu, self.dt)
        Q_lqr = self.Q
        R_lqr = self.R
        P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
        btp = np.dot(B.T, P)
        self.lqr_gain = -np.dot(np.linalg.inv(self.R + np.dot(btp, B)), np.dot(btp, A))
        self.discrete_dfdx = A
        self.discrete_dfdu = B

    def set_gp_dynamics_func(self):
        """Updates symbolic dynamics.

        With actual control frequency, initialize GP model and add to the combined dynamics.

        """
        self.setup_prior_dynamics()
        # Compute the probabilistic constraint inverse CDF according to section III.D.b in Hewing 2019.
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1/self.model.nx - (self.prob + 1)/(2*self.model.nx)))
        # self.create_sparse_GP_machinery()

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        """Converts trajectory data for GP trianing.
        
        Args:
            x_seq (list): state sequence of np.array (nx,). 
            u_seq (list): action sequence of np.array (nu,). 
            x_next_seq (list): next state sequence of np.array (nx,). 
            
        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).

        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        x_pred_seq = self.prior_dynamics_func(x0=x_seq.T - self.prior_ctrl.X_LIN[:, None],
                                               p=u_seq.T - self.prior_ctrl.U_LIN[:,None])['xf'].toarray()
        targets = (x_next_seq.T - (x_pred_seq+self.prior_ctrl.X_LIN[:,None])).transpose()  # (N, nx).
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
        return inputs, targets

    def precompute_probabilistic_limits(self,
                                        print_sets=True
                                        ):
        """This updates the constraint value limits to account for the uncertainty in the dynamics rollout.

        Args:
            print_sets (bool): True to print out the sets for debugging purposes.

        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        state_covariances = np.zeros((self.T+1, nx, nx))
        input_covariances = np.zeros((self.T, nu, nu))

        # Initilize lists for the tightening of each constraint.
        state_constraint_set = []
        for state_constraint in self.constraints.state_constraints:
            state_constraint_set.append(np.zeros((state_constraint.num_constraints, T+1)))

        input_constraint_set = []
        for input_constraint in self.constraints.input_constraints:
            input_constraint_set.append(np.zeros((input_constraint.num_constraints, T)))

        if self.x_prev_mu is not None and self.u_prev is not None and self.x_prev_cov is not None:

            cov_x = np.diag([self.initial_rollout_std**2]*nx)
            cov_u = np.zeros((nu, nu))

            for t in range(T):

                state_covariances[t] = cov_x
                input_covariances[t] = cov_u                

                # Loop through input constraints and tighten by the required ammount.
                for ui, input_constraint in enumerate(self.constraints.input_constraints):
                    input_constraint_set[ui][:, t] = -1*self.inverse_cdf * \
                                                    np.absolute(input_constraint.A) @ np.sqrt(np.diag(cov_u))
                    
                for si, state_constraint in enumerate(self.constraints.state_constraints):
                    state_constraint_set[si][:, t] = -1*self.inverse_cdf * \
                                                    np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
                    
                # Compute the next step propogated state covariance using sigma points    
                # Sigma Point Expand
                sigma_points, weights = generate_sigma_points_gaussian( cs.reshape(self.x_prev_mu[:,t],-1,1), cs.diag( self.x_prev_cov[:,t] ), np.zeros((nx,1)), 1.0 )

                j = 0                
                z = cs.vertcat(sigma_points[:,j], self.u_prev[:,t])
                # mu_d_tensor, cov_d_tensor = self.gaussian_process.predict(z, return_pred=False)
                # cov_d = cov_d_tensor.detach().numpy()
                # mu_d = mu_d_tensor.detach().numpy()
                pred = self.gaussian_process.casadi_predict(z=z)
                cov_d = pred["covariance"]
                mu_d = pred["mean"]
                cov = self.Bd @ cov_d @ self.Bd.T
                mu = self.Bd @ mu_d.reshape((nx,1)) + cs.reshape(self.prior_dynamics_func(x0=cs.reshape(sigma_points[:,j]-self.prior_ctrl.X_LIN[:,None],6,1),
                                                        p=cs.reshape(self.u_prev[:, t],2,1)-self.prior_ctrl.U_LIN[:,None])['xf'], 6,1) + \
                                cs.reshape(self.prior_ctrl.X_LIN[:,None], 6,1)
                root_term = get_ut_cov_root_diagonal(cov) 
                new_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, np.zeros((nx,1)), 1.0 )
                new_weights = temp_weights * weights[:,j]                    
                for j in range(1,2*nx+1):
                    z = cs.vertcat(sigma_points[:,j], self.u_prev[:,t])
                    # mu_d_tensor, cov_d_tensor = self.gaussian_process.predict(z[None,:], return_pred=False)
                    # cov_d = cov_d_tensor.detach().numpy()
                    # mu_d = mu_d_tensor.detach().numpy()
                    pred = self.gaussian_process.casadi_predict(z=z)
                    cov_d = pred["covariance"]
                    mu_d = pred["mean"]
                    cov = self.Bd @ cov_d @ self.Bd.T
                    mu = self.Bd @ mu_d.reshape((nx,1)) + cs.reshape(self.prior_dynamics_func(x0=cs.reshape(sigma_points[:,j]-self.prior_ctrl.X_LIN[:,None],6,1),
                                                            p=cs.reshape(self.u_prev[:, t],2,1)-self.prior_ctrl.U_LIN[:,None])['xf'], 6,1) + \
                                    cs.reshape(self.prior_ctrl.X_LIN[:,None], 6,1)
                    root_term = get_ut_cov_root_diagonal(cov)   
                    temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, np.zeros((nx,1)), 1.0 )
                    new_points = cs.hcat([ new_points, temp_points ])
                    new_weights = cs.hcat([ new_weights, temp_weights * weights[:,j]  ])  

                # Sigma Point compress
                mu_x, cov_x = get_mean_cov( new_points, new_weights )
                   
            # Udate Final covariance.
            for si, state_constraint in enumerate(self.constraints.state_constraints):
                state_constraint_set[si][:,-1] = -1 * self.inverse_cdf * \
                                                np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
            state_covariances[-1] = cov_x
        print(f"state covarainces: {state_covariances}")
        if print_sets:
            print("Probabilistic State Constraint values along Horizon:")
            print(state_constraint_set)
            print("Probabilistic Input Constraint values along Horizon:")
            print(input_constraint_set)
        self.results_dict['input_constraint_set'].append(input_constraint_set)
        self.results_dict['state_constraint_set'].append(state_constraint_set)
        self.results_dict['state_horizon_cov'].append(state_covariances)
        self.results_dict['input_horizon_cov'].append(input_covariances)
        return state_constraint_set, input_constraint_set

    @staticmethod
    def controller(params, X, x_goal, A, bc):

            p = jnp.array([ X[0,0], X[2,0] ]).reshape(-1,1)
            pdot = jnp.array([ X[1,0], X[3,0] ]).reshape(-1,1)

            kx = params[0]
            kv = params[1]
            krx = params[2]
            kR = params[3] 
            kRv = params[4]
            
            # attraction           
            ad_attraction = - kx * (p - jnp.array([x_goal[0,0], x_goal[2,0]]).reshape(-1,1)) - kv * pdot 

            # Repulsion
            Ac = jnp.array([A[0,0], A[0,2]]).reshape(-1,1)
            Ac = Ac / jnp.linalg.norm(Ac) # -Ac is the direction of improving gradient
            ad_repulsion = - krx * Ac  * 1.0 * jnp.tanh( 0.01 / (bc.reshape(-1,1) - A @ X)  ) 
            # pdb.set_trace()
            #Based on perfect knowledge of dynamics
            M = 0.027  #0.027 # 1.0
            I = 1.4e-5  #0.5
            g = 9.80
            L = 0.028

            # toal acceleration
            ad = ad_attraction + ad_repulsion
            target_thrust = ad + jnp.array([ 0, M*g ]).reshape(-1,1)
            theta_d = jnp.arctan2( target_thrust[0,0], target_thrust[1,0] )

            
            scalar_thrust = target_thrust.T @ jnp.array([ jnp.sin(X[4,0]), jnp.cos(X[4,0]) ])
            theta_ddot = - kR * (X[4,0]-theta_d) - kRv * X[5,0]
            moment = I * theta_ddot

            # moment = I * 0.001

            T2 = 0.5 * ( scalar_thrust + moment / L )
            T1 = 0.5 * ( scalar_thrust - moment / L )
            
            # pdb.set_trace()
            action = jnp.array([T1[0], T2[0]])
            action = jnp.clip( jnp.array( [T1[0], T2[0]] ), 0, 0.2)

            # action = jnp.clip(action, 0.0, None )
            # action = action / jnp.max( jnp.array([jnp.linalg.norm(action), 0.01]) ) * 0.2
            # print(f"action: {action}")
            # action = 0.2 * jnp.tanh(action)

            # print(f"theta: {X[4,0]}, thetad: {theta_d}")

            return action

    def setup_sigma_point_expand(self):
        dt = self.dt
        n, N = 6, 13

        @jit
        def sigma_point_expand(sigma_points, weights, params, A, B, x_eq, u_eq, X_GOAL, consA, consb):

            new_points = jnp.zeros((n*(2*n+1),N))
            new_weights = jnp.zeros((2*n+1,N))

            def body(i, inputs):
                new_points, new_weights = inputs        
                u = FORESEE_CBF_QP_COV.controller(params, sigma_points[:,i].reshape(-1,1), X_GOAL, consA, consb).reshape(-1,1)
                z = jnp.append( sigma_points[:,i].reshape(-1,1), u, axis=0 )
                pred_mu, next_state_cov = self.gaussian_process.jax_predict(z=z)
                next_state_cov_root = jnp.zeros((6,6)) #get_ut_cov_root_diagonal(next_state_cov)    
                next_state_mu = (A @ (sigma_points[:,i].reshape(-1,1) - x_eq) + B @ (u - u_eq))*dt + x_eq + pred_mu                       
                temp_points, temp_weights = generate_sigma_points_gaussian( next_state_mu, next_state_cov_root, jnp.zeros((n,1)), 1.0 )
                new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
                new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   
                return new_points, new_weights
            return_points, return_weights = lax.fori_loop(0, N, body, (new_points, new_weights))

            # because return points are all new sigma points of a point in single column. required because JAX inplace operator requires index start/end/gap to be the same
            return return_points.reshape((n, N*(2*n+1)), order='F'), return_weights.reshape((1,N*(2*n+1)), order='F')
        
        return sigma_point_expand

       
    def setup_predict(self, T): #, X, T, A, B, x_eq, u_eq):
            
            dt = self.dt

            @jit
            def predict_future(params, X, A, B, x_eq, u_eq, X_GOAL, consA, consb):             
                sigma_points, weights = generate_sigma_points_gaussian( X, jnp.zeros((6,6)), jnp.zeros((6,1)), 1.0 )   
                mus = jnp.zeros((6, T+1))
                covs = jnp.zeros((6,T+1))         
                def body(i, inputs):
                    sigma_points, weights, mus, covs = inputs
                    expanded_sigma_points, expanded_weights = FORESEE_CBF_QP_COV.sigma_point_expand( sigma_points, weights, params, A, B, x_eq, u_eq, X_GOAL, consA, consb )
                    mu, cov, sigma_points, weights = sigma_point_compress( expanded_sigma_points, expanded_weights )
                    get_mean_cov(sigma_points, weights)
                    mus = mus.at[:,i+1].set( mu[:,0] )
                    covs = covs.at[:,i+1].set( jnp.diag(cov) )
                    return sigma_points, weights, mus, covs        
                mus, covs, final_points, final_weights = lax.fori_loop( 0, T, body, (sigma_points, weights, mus, covs))
                return mus, covs
            return predict_future
    
    @staticmethod
    def reward(Xs, x_goal):
        scale = 1.0
        pos_error = 2 * ( jnp.sum( jnp.square(scale * (Xs[0,:]-x_goal[0,0]) ) ) + jnp.sum( jnp.square(scale * (Xs[2,:]-x_goal[1,0]) ) ) )
        pos_error_terminal = 8 * ( jnp.sum( jnp.square(scale * (Xs[0,-1]-x_goal[0,0]))  ) + jnp.sum( jnp.square(scale * (Xs[2,-1]-x_goal[1,0]) ) ) )
        vel_error = 3000 * ( jnp.sum( jnp.square( scale * Xs[1,:]) ) + jnp.sum( jnp.square(scale * Xs[2,:]) ) )
        # theta_error = 1 * jnp.sum( jnp.square(scale * (Xs[4,:]-x_goal[4,0]) ) )
        return pos_error + vel_error  + pos_error_terminal #+ theta_error
        



    def setup_gp_optimizer(self):
        """Sets up nonlinear optimization problem including cost objective, variable bounds and dynamics constraints.

        """
        FORESEE_CBF_QP_COV.sigma_point_expand = self.setup_sigma_point_expand()
        FORESEE_CBF_QP_COV.predict = self.setup_predict(5)#self.T)

        FORESEE_CBF_QP_COV.reward_func = lambda params, X, A, B, x_eq, u_eq, X_GOAL, consA, consb: FORESEE_CBF_QP_COV.reward( FORESEE_CBF_QP_COV.predict(params, X, A, B, x_eq, u_eq, X_GOAL, consA, consb)[0], X_GOAL )

        FORESEE_CBF_QP_COV.predict_grad = jit( value_and_grad( FORESEE_CBF_QP_COV.reward_func, 0 ) )
        return

        ########################################################



    def select_action_with_gp(self,
                              obs
                              ):
        """Solves nonlinear MPC problem to get next action.

         Args:
             obs (np.array): current state/observation.

         Returns:
             np.array: input/action to the task/env.

         """
        
        print(f"time step: {self.dt}")
        x_goal = self.env.X_GOAL.reshape(-1,1) #jnp.array([-1.0, 0.0]).reshape(-1,1)  #jnp.array([-0.7, 0.5]).reshape(-1,1)  # 
        # Select current action
        z = jnp.append( obs.reshape(-1,1), jnp.zeros((2,1)), axis=0 )
        self.gaussian_process.jax_predict(z=z)     
        action = FORESEE_CBF_QP_COV.controller(self.params, obs.reshape(-1,1), x_goal, self.constraints.state_constraints[1].A, self.constraints.state_constraints[1].b)
        
        
        # Update parameters
        nx, nu = self.model.nx, self.model.nu
        T = self.T

        A, B, dt = self.prior_ctrl.dfdx, self.prior_ctrl.dfdu, self.dt
        x_eq = self.prior_ctrl.X_LIN[:,None].reshape(-1,1)
        u_eq = self.prior_ctrl.U_LIN[:,None].reshape(-1,1)

        # print(f"A; {A} \n B:{B}, \n x_eq: {x_eq}, \n u_eq: {u_eq}")
        # exit()
        
        reward, grads = FORESEE_CBF_QP_COV.predict_grad( jnp.copy(self.params), obs.reshape(-1,1), A, B, x_eq, u_eq, x_goal, self.constraints.state_constraints[1].A, self.constraints.state_constraints[1].b )
        print(f"rewards: {reward}, gras:{grads}")

        t0 = time.time()
        
        if self.adapt:
            print(f"HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            for i in range(self.num_adapt_iterations):
                states, actions = FORESEE_CBF_QP_COV.predict(jnp.copy(self.params), obs.reshape(-1,1), A, B, x_eq, u_eq, x_goal, self.constraints.state_constraints[1].A, self.constraints.state_constraints[1].b)
                reward, grads = FORESEE_CBF_QP_COV.predict_grad( jnp.copy(self.params), obs.reshape(-1,1), A, B, x_eq, u_eq, x_goal, self.constraints.state_constraints[1].A, self.constraints.state_constraints[1].b )
                # print(f"reward: {reward}, grad: {grads}")
                # exit()
                # pdb.set_trace()
                grads = np.asarray(grads)
                lr = 0.0001 # unstable with 0.001. need atleast 0.0001
                self.params[0] = np.clip( self.params[0] - lr * np.clip( grads[0], -1.0, 1.0 ), 0.0, None )
                self.params[1] = np.clip( self.params[1] - lr * np.clip( grads[1], -1.0, 1.0 ), 0.0, None )
                self.params[2] = np.clip( self.params[2] - lr * np.clip( grads[2], -1.0, 1.0 ), 0.0, None )
            print(f"************************************************* time: {time.time()-t0} *********************************************")
            print(f"*************** params: {self.params}")
        
        return action, jnp.copy(self.params)         

    def learn(self,
              input_data=None,
              target_data=None,
              gp_model=None,
              plot=False
              ):
        """Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            plot (bool): to plot validation trajectories or not.

        Returns:
            training_results (dict): Dictionary of the training results.

        """
        if gp_model is None:
            gp_model = self.gp_model_path
        self.prior_ctrl.remove_constraints(self.prior_ctrl.additional_constraints)
        self.reset()
        if self.online_learning:
            input_data = np.zeros((self.train_iterations, len(self.input_mask)))
            target_data = np.zeros((self.train_iterations, len(self.target_mask)))
        get_new_data = False
        if get_new_data:
            if input_data is None and target_data is None:
                train_inputs = []
                train_targets = []
                train_info = []

                ############
                # Use Latin Hypercube Sampling to generate states withing environment bounds.
                lhs_sampler = Lhs(lhs_type='classic', criterion='maximin')
                limits = [(self.env.INIT_STATE_RAND_INFO[key].low, self.env.INIT_STATE_RAND_INFO[key].high) for key in
                        self.env.INIT_STATE_RAND_INFO]
                # todo: parameterize this if we actually want it.
                num_eq_samples = 0
                samples = lhs_sampler.generate(limits,
                                            self.train_iterations + self.validation_iterations - num_eq_samples,
                                            random_state=self.seed)
                # todo: choose if we want eq samples or not.
                delta = 0.01
                eq_limits = [(self.prior_ctrl.X_LIN[eq]-delta, self.prior_ctrl.X_LIN[eq]+delta) for eq in range(self.model.nx)]
                if num_eq_samples > 0:
                    eq_samples = lhs_sampler.generate(eq_limits, num_eq_samples, random_state=self.seed)
                    #samples = samples.append(eq_samples)
                    init_state_samples = np.array(samples + eq_samples)
                else:
                    init_state_samples = np.array(samples)
                input_limits = np.vstack((self.constraints.input_constraints[0].lower_bounds,
                                        self.constraints.input_constraints[0].upper_bounds)).T
                input_samples = lhs_sampler.generate(input_limits,
                                                    self.train_iterations + self.validation_iterations,
                                                    random_state=self.seed)
                input_samples = np.array(input_samples) # not being used currently
                #seeds = self.env.np_random.randint(0,99999, size=self.train_iterations + self.validation_iterations)
                seeds = self.env.np_random.integers(0,99999, size=self.train_iterations + self.validation_iterations)
                for i in range(self.train_iterations + self.validation_iterations):
                    # For random initial state training.
                    init_state = init_state_samples[i,:]
                    # Collect data with prior controller.
                    run_env = self.env_func(init_state=init_state, randomized_init=False, seed=int(seeds[i]))
                    episode_results = self.prior_ctrl.run(env=run_env, max_steps=1)
                    run_env.close()
                    x_obs = episode_results['obs'][-3:,:]
                    u_seq = episode_results['action'][-1:,:]
                    run_env.close()
                    x_seq = x_obs[:-1,:]
                    x_next_seq = x_obs[1:,:]
                    train_inputs_i, train_targets_i = self.preprocess_training_data(x_seq, u_seq, x_next_seq)
                    train_inputs.append(train_inputs_i)
                    train_targets.append(train_targets_i)
                ###########
            else:
                train_inputs = input_data
                train_targets = target_data
            # assign all data
            train_inputs = np.vstack(train_inputs)
            train_targets = np.vstack(train_targets)
            self.data_inputs = train_inputs
            self.data_targets = train_targets
            train_idx, test_idx = train_test_split(
                                    #list(range(self.train_iterations + self.validation_iterations)),
                                    list(range(train_inputs.shape[0])),
                                    test_size=self.validation_iterations/(self.train_iterations+self.validation_iterations),
                                    random_state=self.seed
                                    )
            train_inputs = self.data_inputs[train_idx, :]
            train_targets = self.data_targets[train_idx, :]
            self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
            test_inputs = self.data_inputs[test_idx, :]
            test_targets = self.data_targets[test_idx, :]
            self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}

            with open('gp_train_data.npy', 'wb') as f:
                np.save(f, train_inputs)
                np.save(f, train_targets)
                np.save(f, test_inputs)
                np.save(f, test_targets)

        else:
            with open('gp_train_data.npy', 'rb') as f:
                train_inputs = np.load(f)
                train_targets = np.load(f)
                test_inputs = np.load(f)
                test_targets = np.load(f)


        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()


        plot = False
        if plot:
            init_state = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            valid_env = self.env_func(init_state=init_state,
                                      randomized_init=False)
            validation_results = self.prior_ctrl.run(env=valid_env,
                                                     max_steps=40)
            valid_env.close()
            x_obs = validation_results['obs']
            u_seq = validation_results['action']
            x_seq = x_obs[:-1, :]
            x_next_seq = x_obs[1:, :]
        # Define likelihood.
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        self.gaussian_process = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                                     likelihood,
                                                     len(self.target_mask),
                                                     input_mask=self.input_mask,
                                                     target_mask=self.target_mask,
                                                     normalize=self.normalize_training_data
                                                     )
        # print(f"{len(self.gaussian_process.gp_list)}")
        # exit()
        if gp_model:
            self.gaussian_process.init_with_hyperparam(train_inputs_tensor,
                                                       train_targets_tensor,
                                                       gp_model)
        else:
            # Train the GP.
            self.gaussian_process.train(train_inputs_tensor,
                                        train_targets_tensor,
                                        test_inputs_tensor,
                                        test_targets_tensor,
                                        n_train=self.optimization_iterations,
                                        learning_rate=self.learning_rate,
                                        gpu=self.use_gpu,
                                        dir=self.output_dir)
        # Plot validation.
        if plot:
            validation_inputs, validation_targets = self.preprocess_training_data(x_seq, u_seq, x_next_seq)
            fig_count = 0
            fig_count = self.gaussian_process.plot_trained_gp(torch.Tensor(validation_inputs).double(),
                                                              torch.Tensor(validation_targets).double(),
                                                              fig_count=fig_count)
        self.set_gp_dynamics_func()
        self.setup_gp_optimizer()
        self.prior_ctrl.add_constraints(self.prior_ctrl.additional_constraints)
        self.prior_ctrl.reset()
        # Collect training results.
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        try:
            training_results['info'] = train_info
        except UnboundLocalError:
            training_results['info'] = None
        return training_results

    def select_action(self,
                      obs
                      ):
        """Select the action based on the given observation.

        Args:
            obs (np.array): current observed state.

        Returns:
            action (np.array): desired policy action.

        """
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            if(self.last_obs is not None and self.last_action is not None and self.online_learning):
                print("[ERROR]: Not yet supported.")
                exit()
            t1 = time.perf_counter()
            action, params = self.select_action_with_gp(obs)
            t2 = time.perf_counter()
            print("GP SELECT ACTION TIME: %s" %(t2 - t1))
            self.last_obs = obs
            self.last_action = action
        return action, params

    def close(self):
        """Clean up.

        """
        self.env_training.close()
        self.env.close()

    def reset_results_dict(self):
        """

        """
        "Result the results_dict before running."
        super().reset_results_dict()
        self.results_dict['input_constraint_set'] = []
        self.results_dict['state_constraint_set'] = []
        self.results_dict['state_horizon_cov'] = []
        self.results_dict['input_horizon_cov'] = []
        # self.results_dict['gp_mean_eq_pred'] = []
        self.results_dict['gp_pred'] = []
        self.results_dict['linear_pred'] = []

    def reset(self):
        """Reset the controller before running.

        """
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        if self.gaussian_process is not None:
            self.set_gp_dynamics_func()
            # CasADi optimizer.
            self.setup_gp_optimizer()
        self.prior_ctrl.reset()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev_mu = None
        self.x_prev_cov = None
        # self.weights_prev = None
        self.u_prev = None


# action = jnp.array([0, 0.2])
# (Pdb) continue
# GP SELECT ACTION TIME: 68.7822292369965
# 0 -th step.
# action: [0.  0.2]
# obs: [-9.90075468e-01  3.16756707e-01  4.10690396e-03  1.86162004e-01
#   8.42593127e-01  5.93979960e+00]



# GP SELECT ACTION TIME: 29.661592380012735
# 0 -th step.
# action: [0.2 0. ]
# obs: [-1.00992453e+00 -3.16756707e-01  4.10690396e-03  1.86162004e-01
#  -8.42593127e-01 -5.93979960e+00]