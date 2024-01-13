import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import jax.numpy as jnp
from jax import jit, lax
# import numpy as np

from jax import config
config.update("jax_enable_x64", True)

# @jit
def step(x,u,dt):
    return x+u*dt

def dynamics_step( base_term, state_dot, dt ):
    next_state = base_term + state_dot * dt
#     print(f"next_state:{next_state}")
    return next_state

# assume a single control input
# assume this is true dynamics
def dynamics_xdot_noisy(state, action):
    xdot = jnp.array([ state[0,0]**2, state[1,0]**2 ]).reshape(-1,1)
    cov = jnp.zeros((2,2))
    # error_square = 0.01 + 0.1 * jnp.square(xdot) # /2  #never let it be 0!!!!
    # cov = jnp.diag( error_square[:,0] )
    return xdot, cov

@jit
def get_mean( sigma_points, weights ):
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

@jit
def get_mean_cov(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    cov = jnp.diag(jnp.sum(centered_points**2 * weights[0], axis=1))
    return mu, cov

def get_ut_cov_root_diagonal(cov):
    # return jnp.zeros((4,4))
    offset = 0.000  # TODO: make sure not zero here
    root_term = jnp.diag( jnp.diagonal(cov)+offset  )
    return root_term
    
    #root0 = jnp.sqrt((offset+cov[0,0]))
    #root1 = jnp.sqrt((offset+cov[1,1]))
    #root_term = jnp.diag( jnp.array([root0, root1]) )
    #return root_term


@jit
def get_mean_cov_skew_kurt( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)    
    centered_points = sigma_points - mu    
    cov = jnp.diag(jnp.sum(centered_points**2 * weights[0], axis=1))
    
    skewness = jnp.sum(centered_points**3 * weights[0], axis=1) #/ cov[0,0]**(3/2) # for scipy    
    kurt = jnp.sum(centered_points**4 * weights[0], axis=1)# / cov[0,0]**(4/2)  # -3 # -3 for scipy
    return mu, cov, skewness.reshape(-1,1), kurt.reshape(-1,1)

@jit
def get_mean_cov_skew_kurt_for_generation( sigma_points, weights ):
    # mean
    weighted_points = sigma_points * weights[0]
    mu = jnp.sum( weighted_points, 1 ).reshape(-1,1)    
    centered_points = sigma_points - mu    
    cov = jnp.diag(jnp.sum(centered_points**2 * weights[0], axis=1))
    
    skewness_temp = jnp.sum(centered_points**3 * weights[0], axis=1) #/ cov[0,0]**(3/2) # for scipy    
    skewness = skewness_temp[0] / cov[0,0]**(3/2)
    skewness = jnp.append(skewness, skewness_temp[1] / cov[1,1]**(3/2))
    skewness = jnp.append(skewness, skewness_temp[2] / cov[2,2]**(3/2))
    skewness = jnp.append(skewness, skewness_temp[3] / cov[3,3]**(3/2))
    kurt_temp = jnp.sum(centered_points**4 * weights[0], axis=1)# / cov[0,0]**(4/2)  # -3 # -3 for scipy
    kurt = kurt_temp[0]/cov[0,0]**(4/2)
    kurt = jnp.append(kurt, kurt_temp[1]/cov[1,1]**(4/2))
    kurt = jnp.append(kurt, kurt_temp[2]/cov[2,2]**(4/2))
    kurt = jnp.append(kurt, kurt_temp[3]/cov[3,3]**(4/2))

    return mu, cov, skewness.reshape(-1,1), kurt.reshape(-1,1)

@jit
def generate_sigma_points_gaussian( mu, cov_root, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    alpha = 1.0
    beta = 0.0#2.0#2.0 # optimal for gaussian
    k = 1.0
    Lambda = alpha**2 * ( n+k ) - n
    
    points0 = base_term + mu * factor
    points1 = base_term + (mu + jnp.sqrt(n+Lambda) * cov_root)*factor
    points2 = base_term + (mu - jnp.sqrt(n+Lambda) * cov_root)*factor
    # print(f"{points0}, {points1}, {points2}")
    
    weights0 = jnp.array([[ 1.0*Lambda/(n+Lambda) ]])
    weights1 = jnp.ones((1,n)) * 1.0/(n+Lambda)/2.0
    weights2 = jnp.ones((1,n)) * 1.0/(n+Lambda)/2.0
    # print(f"n: {n} \n {weights0}, {weights1}, {weights2}")
    new_points = jnp.concatenate((points0, points1, points2), axis=1)
    new_weights = jnp.concatenate((weights0, weights1, weights2), axis=1)
    
    return new_points, new_weights    

@jit
def generate_sigma_points_gaussian_GenUT( mu, cov_root, skewness, kurt, base_term, factor ):
    n = mu.shape[0]     
    N = 2*n + 1 # total points
    u = 0.5 * ( - skewness + jnp.sqrt( 4 * kurt - 3 * ( skewness )**2 ) )
    v = u + skewness

    w2 = (1.0 / v) / (u+v)
    w1 = (w2 * v) / u
    w0 = jnp.array([1 - jnp.sum(w1) - jnp.sum(w2)])
    
    U = jnp.diag(u[:,0])
    V = jnp.diag(v[:,0])
    points0 = base_term + mu * factor
    points1 = base_term + (mu - cov_root @ U) * factor
    points2 = base_term + (mu + cov_root @ V) * factor
    new_points = jnp.concatenate( (points0, points1, points2), axis=1 )
    new_weights = jnp.concatenate( (w0.reshape(-1,1), w1.reshape(1,-1), w2.reshape(1,-1)), axis=1 )

    return new_points, new_weights

# @jit
# def sigma_point_expand(sigma_points, weights, control, dt):
   
#     n, N = sigma_points.shape   
    
#     # new_points = jnp.zeros((n,N*(2*n+1)))
#     # new_weights = jnp.zeros((1,N*(2*n+1)))

#     # because Jax cannot do .at[start, stop] operation without having fixed start/stop/step ..
#     new_points = jnp.zeros((n*(2*n+1),N))
#     new_weights = jnp.zeros((2*n+1,N))
    
#     def body(i, inputs):
#         new_points, new_weights = inputs        
#         mu, cov = dynamics_xdot_noisy(sigma_points[:,i].reshape(-1,1), control.reshape(-1,1))
#         root_term = get_ut_cov_root_diagonal(cov)           
#         temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, sigma_points[:,i].reshape(-1,1), dt )
#         new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
#         new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   
#         return new_points, new_weights
#     return_points, return_weights = lax.fori_loop(0, N, body, (new_points, new_weights))
#     return return_points.reshape((n, N*(2*n+1)), order='F'), return_weights.reshape((1,N*(2*n+1)), order='F')
        

@jit
def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = jnp.zeros((mu.shape))
    new_points, new_weights = generate_sigma_points_gaussian( mu, cov_root_term, base_term, jnp.array([1.0]) )
    return mu, cov, new_points, new_weights

@jit
def sigma_point_compress_GenUT( sigma_points, weights ):
    mu, cov, skewness, kurt = get_mean_cov_skew_kurt_for_generation( sigma_points, weights )
    # print(f"mu:{mu}, cov:{cov}, skewness:{skewness}, kurtosis:{kurt}")
    cov_root_term = get_ut_cov_root_diagonal( cov )  
    base_term = jnp.zeros((mu.shape))
    return generate_sigma_points_gaussian_GenUT( mu, cov_root_term, skewness, kurt, base_term, jnp.array([1.0]) )

@jit
def foresee_propagate_GenUT( sigma_points, weights, action, dt ):
    
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action, dt )
    compressed_sigma_points, compressed_weights = sigma_point_compress_GenUT(expanded_sigma_points, expanded_weights)
    return compressed_sigma_points, compressed_weights

@jit
def foresee_propagate( sigma_points, weights, action, dt ):

    #Expansion Layer
    expanded_sigma_points, expanded_weights = sigma_point_expand( sigma_points, weights, action, dt )
    compressed_sigma_points, compressed_weights = sigma_point_compress(expanded_sigma_points, expanded_weights)
    return compressed_sigma_points, compressed_weights

# @jit 
# def state_predict(init_state):
#     T = 30

#     sigma_points, weights = generate_sigma_points_gaussian( init_state, jnp.zeros((init_state.shape[0], init_state.shape[0])), jnp.zeros(init_state.shape), 1.0  )
#     def body(i, inputs):
#         state = inputs
#         control = controller()
#     return 
