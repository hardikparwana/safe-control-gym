"""This script runs the GP-MPC experiment in our Annual Reviews article.

See Figure 6 in https://arxiv.org/pdf/2108.06266.pdf.

"""

# quad config: https://github.com/utiasDSL/safe-control-gym/blob/83fae93172782f7c6e98063da0b2425d3f741f1b/safe_control_gym/envs/gym_pybullet_drones/quadrotor.yaml#L17
# quad model: https://github.com/utiasDSL/safe-control-gym/blob/83fae93172782f7c6e98063da0b2425d3f741f1b/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import munch
import numpy as np
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory

import jax.numpy as jnp

import pdb

def reward(Xs, x_goal):
    scale = 1.0 #0.00001
    pos_error = 20 * ( jnp.sum( jnp.square(scale * (Xs[0,:]-x_goal[0,0]) ) ) + jnp.sum( jnp.square(scale * (Xs[2,:]-x_goal[1,0]) ) ) )
    pos_error_terminal = 80 * ( jnp.sum( jnp.square(scale * (Xs[0,-1]-x_goal[0,0]))  ) + jnp.sum( jnp.square(scale * (Xs[2,-1]-x_goal[1,0]) ) ) )
    vel_error = 2 * ( jnp.sum( jnp.square( scale * Xs[1,:]) ) + jnp.sum( jnp.square(scale * Xs[2,:]) ) )
    # theta_error = 1 * jnp.sum( jnp.square(scale * (Xs[4,:]-x_goal[4,0]) ) )
    return pos_error + vel_error  + pos_error_terminal #+ theta_error


def plot_xz_comparison_diag_constraint(prior_run,
                                       run,
                                       init_ind,
                                       dir=None
                                       ):
    """

    """
    state_inds = [0,2]
    goal = [0, 1]
    ax = plot_2D_comparison_with_prior(state_inds, prior_run, run, goal, init_ind, dir=dir)
    limit_vals = np.array([[-2.1, -1.0],
                           [2.0, 3.1]])
    ax.plot(limit_vals[:,0], limit_vals[:,1], 'r-', label='Limit')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    if dir is not None:
        np.savetxt(os.path.join(dir, 'limit.csv'), limit_vals, delimiter=',', header='x_limit,y_limit')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Z Position [m]')
    ax.set_xlim([-1.2, 0.1])
    ax.set_ylim([-0.05, 1.1])
    ax.set_box_aspect(0.5)
    plt.tight_layout()
    plt.savefig(dir+"comparison.png")

    x_goal = jnp.array([0, 1.0]).reshape(-1,1)
    rewards = reward( run.obs.T, x_goal )

    fig2, ax2 = plt.subplots(2,2)
    ax2[0,0].plot( run.obs[:,4], 'k*-', label=r'$\theta$' )
    ax2[0,0].legend()

    # fig3, ax3 = plt.subplots()
    ax2[0,1].plot( run.action[:,0], 'r-*', label='T1' )
    ax2[0,1].plot( run.action[:,1], 'b-*', label='T2' )
    ax2[0,1].legend()

    # fig4, ax4 = plt.subplots()
    ax2[1,0].plot( run.obs[:,0], 'r-*', label='x' )
    ax2[1,0].plot( run.obs[:,2], 'b-*', label='z' )
    ax2[1,0].set_title(f"Reward = {rewards}")
    ax2[1,0].legend()

    # pdb.set_trace()
    # fig5, ax5 = plt.subplots()
    ax2[1,1].plot( run.params[:,0], 'r-*', label='kx' )
    ax2[1,1].plot( run.params[:,1], 'b-*', label='kv' )
    ax2[1,1].plot( run.params[:,2], 'c-*', label='krx' )
    # ax5.plot( run.params[:,3], 'm-*', label='kR' )
    # ax5.plot( run.params[:,4], 'k-*', label='kRv' )
    ax2[1,1].legend()

    # print(f"*********************** INFO ***************************")
    # print(f"Actual reward: { rewards }")
    # print(f"*********************** INFO ***************************")
          
    # fig6, ax6 = plt.subplots()
    # ax6.plot( rewards, 'r-*', label='reward' )
    # ax6.legend()


def plot_2D_comparison_with_prior(state_inds,
                                  prior_run,
                                  crun, goal,
                                  init_ind,
                                  dir=None
                                  ):
    """

    """

    fig, ax = plt.subplots()
    ax.plot(goal[0], goal[1], 'g',
            marker='x',
            markersize=12,
            markeredgewidth=2,
            label='Goal')
    ax.plot(crun.obs[:, state_inds[0]], crun.obs[:, state_inds[1]], '-', label='GP-MPC')

    prior_horizon_states = prior_run.horizon_states[init_ind]
    for i in range(crun.state_horizon_cov[init_ind].shape[0]): # time horizon
        prior_position = prior_horizon_states[state_inds,i]

        if i == 1:
            ax.plot(prior_position[0], prior_position[1], 'm.', label='Linear MPC Prediction horizon')
        else:
            ax.plot(prior_position[0], prior_position[1], 'm.')
        ax.annotate(str(i), prior_position)

    xx = init_ind
    while (xx<len(crun.state_horizon_cov)):
        init_ind = xx

        horizon_cov = crun.state_horizon_cov[init_ind]
        horizon_states = crun.horizon_states[init_ind]
        # pdb.set_trace()
        
        
        
        final_ind = -1
        # ax.plot(prior_run.obs[:,state_inds[0]], prior_run.obs[:,state_inds[1]], '-', label='Linear MPC')
        
        if dir is not None:
            np.savetxt(os.path.join(dir, 'goal.csv'), np.array([goal]),  delimiter=',',header='x_goal,y_goal')
            np.savetxt(os.path.join(dir,'linear_mpc.csv'), prior_run.obs[:,state_inds],  delimiter=',',header='x_linearmpc,y_linearmpc')
            np.savetxt(os.path.join(dir,'gp_mpc.csv'), crun.obs[:,state_inds],  delimiter=',',header='x_gpmpc,y_gpmpc')
            # pdb.set_trace()
            np.savetxt(os.path.join(dir,'gp_mpc_horizon.csv'), horizon_states[state_inds].T, delimiter=',',
                    header='x_gpmpc-horizon,y_gpmpc-horizon')
            # np.savetxt(os.path.join(dir, 'linear_mpc_horizon.csv'), prior_horizon_states[state_inds].T, delimiter=',',
            #         header='x_linear-horizon,y_linear-horizon')
            run_ellipse_data = np.zeros((horizon_cov.shape[0], 2+1+1+1))
        for i in range(horizon_cov.shape[0]): # time horizon
            cov = np.zeros((2, 2))
            cov[0,0] = horizon_cov[i, state_inds[0], state_inds[0]]
            cov[0,1] = horizon_cov[i, state_inds[0], state_inds[1]]
            cov[1,0] = horizon_cov[i, state_inds[1], state_inds[0]]
            cov[1,1] = horizon_cov[i, state_inds[1], state_inds[1]]
            print(f"***************** PRINT INFO ************************")
            print(f"cov: {cov}")
            print(f"***************** PRINT INFO ************************")
            position = horizon_states[state_inds, i]
            
            if i == 1:
                # ax.plot(position[0], position[1], 'k.', label='GP-MPC Prediction horizon')
                ax.plot(position[0], position[1], '.', label='GP-MPC Prediction horizon')
                pos, major_axis_length, minor_axis_length, alpha = add_2d_cov_ellipse(position, cov, ax, legend=True)
                if dir is not None:
                    run_ellipse_data[i,:2] = pos
                    run_ellipse_data[i,2] = major_axis_length
                    run_ellipse_data[i,3] = minor_axis_length
                    run_ellipse_data[i,4] = alpha
            else:
                ax.plot(position[0], position[1], 'k.')
                pos, major_axis_length, minor_axis_length, alpha = add_2d_cov_ellipse(position, cov, ax)
                if dir is not None:
                    run_ellipse_data[i, :2] = pos
                    run_ellipse_data[i, 2] = major_axis_length
                    run_ellipse_data[i, 3] = minor_axis_length
                    run_ellipse_data[i, 4] = alpha
            ax.annotate(str(i), position)
        if dir is not None:
            np.savetxt(os.path.join(dir, 'cov_ellipses.csv'), run_ellipse_data, delimiter=',',
                    header='pos_x,pos_y,major_axis_length,minor_axis_length,alpha')
            
        xx = xx + 5

    ax.set_aspect('equal')
    ax.axis('equal')
    ax.legend()
    return ax


def add_2d_cov_ellipse(position,
                       cov,
                       ax,
                       legend=False
                       ):
    """

    """
    evals, evecs = np.linalg.eig(cov)
    major_axis_ind = np.argmax(evals)
    minor_axis_ind = 0 if major_axis_ind == 1 else 1
    major_eval = evals[major_axis_ind]
    minor_eval = evals[minor_axis_ind]
    major_evec = evecs[:,major_axis_ind]
    minor_evec = evecs[:,minor_axis_ind]
    alpha = np.arctan2(major_evec[1], major_evec[0])*180/np.pi
    # For 95% confidence interval, you must multiply by sqrt 5.991.
    major_axis_length = 2*np.sqrt(5.991*major_eval)
    minor_axis_length = 2*np.sqrt(5.991*minor_eval)
    if legend:
        ellipse = Ellipse(position,
                          major_axis_length,
                          minor_axis_length,
                          angle=alpha,
                          alpha=0.5,
                          label='95% C.I.')
    else:
        ellipse = Ellipse(position,
                          major_axis_length,
                          minor_axis_length,
                          angle=alpha,
                          alpha=0.5)
    ax.add_artist(ellipse)
    return position, major_axis_length, minor_axis_length, alpha


if __name__ == "__main__":
    fac = ConfigFactory()
    fac.add_argument("--train_only", type=bool, default=False, help="True if only training is performed.")
    config = fac.merge()
    # Create environment.
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    # print(f"env_func: {env_func}")
    # exit()
    # Create GP controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    ctrl.reset()
    # Initial state for comparison.
    init_state = {'init_x': -1.0,
                  'init_x_dot': 0.0,
                  'init_z': 0.0,
                  'init_z_dot': 0.0,
                  'init_theta': 0.0,
                  'init_theta_dot': 0.0
                  }
    # Run the prior controller.
    test_env = env_func(init_state=init_state,
                        randomized_init=False)
    prior_results = ctrl.prior_ctrl.run(env=test_env,
                                        max_steps=30)

    # Learn the gp by collecting training points.
    ctrl.learn()
    # input("Press any key to continue.. ")
    print("Press any key to continue.. ")
    if not config.train_only:
        # Run with the learned gp model.

        run_results = ctrl.run(env=test_env,
                               max_steps=30)  #50)
        ctrl.close()
        # Plot the results.
        prior_run = munch.munchify(prior_results)
        run = munch.munchify(run_results)
        plot_xz_comparison_diag_constraint(prior_run, run, 1, dir=config.image_dir)
        plt.show()
