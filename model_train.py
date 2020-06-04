import DGM
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')

# Model parameters

# PDE domain
t_low = 0 + 10e-15    # time lower bound
T = 1 - 10e-15 # time upper bound
x_low = 0.0 + 10e-15  # X lower bound
x_high = 1 - 10e-15   # X upper bound


# Parameters domains
nu_low = 30.0
nu_high = 40.0
gamma_low = -2.0
gamma_high = 2.0
theta_low = 1.0
theta_high = 1.0

# NN parameters
num_layers = 3
nodes_per_layer = 10
learning_rate = 0.0005

# Training parameters
sampling_stages  = 50   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 1000

# Model tensor placeholders

# inputs (time, space domain interior, space domain at initial time)
t_interior_tnsr = tf.placeholder(tf.float64, [None,1])
x_interior_tnsr = tf.placeholder(tf.float64, [None,1])

#initial conditions
t_initial_tnsr = tf.placeholder(tf.float64, [None,1])
x_initial_tnsr = tf.placeholder(tf.float64, [None,1])

#boundary conditions
x_boundary_tnsr = tf.placeholder(tf.float64, [None,1])
x_boundary_last_tnsr = tf.placeholder(tf.float64, [None,1])

#parameters
param_gamma_tnsr = tf.placeholder(tf.float64, [None, 1])
param_theta_tnsr = tf.placeholder(tf.float64, [None, 1])
param_nu_tnsr = tf.placeholder(tf.float64, [None, 1])


def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: PDE domain interior
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    x_interior = np.random.uniform(low=x_low, high=x_high, size=[nSim_interior, 1])

    # Sampler #2: boundary
    x_boundary = np.ones(shape=(nSim_terminal, 1), dtype=np.float64)*(10e-50)
    x_boundary_last =  np.ones(shape=(nSim_terminal, 1), dtype=np.float64)*(1-10e-50) 

    # Sampler #3: initial condition
    t_terminal = np.zeros(shape = (nSim_terminal, 1))
    x_terminal = np.random.uniform(low=0, high=1, size = [nSim_terminal, 1])

    # Sampler #4: parameters interior
    gamma_interior = np.random.uniform(low=gamma_low, high=gamma_high, size=[nSim_interior, 1])
    theta_interior = np.random.uniform(low=theta_low, high=theta_high, size=[nSim_interior, 1])
    nu_interior = np.random.uniform(low=nu_low, high=nu_high, size=[nSim_interior, 1])


    return t_interior, x_interior, t_terminal, x_terminal, x_boundary, x_boundary_last, gamma_interior, theta_interior, nu_interior

def loss(model, t_interior, x_interior, 
                t_terminal, x_terminal,
                x_boundary, x_boundary_last,
                param_gamma, param_theta, param_nu):

    # Loss term #1: PDE
    V = model(t_interior, [x_interior], [param_gamma, param_theta, param_nu])
    V_t = tf.gradients(V, t_interior)[0] #du/dt
    V_s = tf.gradients(V, x_interior)[0] #du/dx
    V_ss = tf.gradients(V_s, x_interior)[0] #d^2u/dx^2

    b = x_interior*(1-x_interior)/(2*param_nu)
    a = param_gamma * x_interior*(1-x_interior)
    diff_V = V_t - b * V_ss + a * V_s

    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term #2: boundary conditions
    boundary_target_payoff = param_nu*param_theta
    boundary_fitted_payoff = model(t_interior, [x_boundary], [param_gamma, param_theta, param_nu])
    boundary_fitted_right_payoff = model(t_interior, [x_boundary_last], [param_gamma, param_theta, param_nu])
    L2_left_boundary = tf.reduce_mean(tf.square(boundary_fitted_payoff - boundary_target_payoff))
    L2_right_boundary = tf.reduce_mean(tf.square(boundary_fitted_right_payoff)) 
    L2 = L2_left_boundary + L2_right_boundary
    
    # Loss term #3: initial condition
    target_payoff =  param_nu * param_theta*(1-tf.exp(-2*param_gamma*(1-x_terminal)))/(1-tf.exp(-2*param_gamma)) 
    fitted_payoff = model(t_terminal, [x_terminal], [param_gamma, param_theta, param_nu])
    L3 = tf.reduce_mean( tf.square(fitted_payoff - target_payoff) )

    return L1, L2, L3

def build_model():
    #3 - number of parameters
    model = DGM.DGMNet(nodes_per_layer, num_layers, 1, 3)

    # loss 
    L1_tnsr, L2_tnsr, L3_tnsr = loss(model, t_interior_tnsr, x_interior_tnsr, 
                                            t_initial_tnsr, x_initial_tnsr, 
                                            x_boundary_tnsr, x_boundary_last_tnsr,
                                            param_gamma_tnsr,
                                            param_theta_tnsr,
                                            param_nu_tnsr)
    loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

    # option value function
    V = model(t_interior_tnsr, [x_interior_tnsr], [param_gamma_tnsr, param_theta_tnsr, param_nu_tnsr])

    # set optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

    # initialize variables
    init_op = tf.global_variables_initializer()

    # open session
    sess = tf.Session()
    sess.run(init_op)

    return sess, loss_tnsr, optimizer

def train_model(sess, loss_tnsr, optimizer):
    # for each sampling stage
    for i in range(sampling_stages):
        
        # sample uniformly from the required regions
        t_interior, x_interior, t_initial, x_initial, x_boundary, x_boundary_last, param_gamma, param_theta, param_nu = sampler(nSim_interior, nSim_terminal)
        # for a given sample, take the required number of SGD steps
        for it in range(steps_per_sample):
            loss,_ = sess.run([loss_tnsr, optimizer],
                                    feed_dict = {t_interior_tnsr:t_interior, x_interior_tnsr:x_interior,
                                    t_initial_tnsr:t_initial, x_initial_tnsr:x_initial, 
                                    x_boundary_tnsr:x_boundary, 
                                    x_boundary_last_tnsr:x_boundary_last,
                                    param_gamma_tnsr: param_gamma, param_theta_tnsr: param_theta, param_nu_tnsr: param_nu})
        t_interior = None
        x_interior = None
        t_initial = None
        x_initial = None
        x_boundary = None
        x_boundary_last = None

        print(f"#{i}/{sampling_stages}: loss: {loss}")


if __name__ == "__main__":
    output_file_name = 'one_pop_example'
    sess, loss_tnsr, optimizer = build_model()
    train_model(sess, loss_tnsr, optimizer)
    saver = tf.train.Saver()
    saver.save(sess, f'./trained_models/{output_file_name}')
    print(f'The model is saved in trained_models/{output_file_name}')