# Copyright (c) 2019 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import warnings
import os

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
from likelihoods.gaussian import Gaussian

from continualgp.het_likelihood import HetLikelihood
from continualgp.continualgp import ContinualGP
from continualgp import util
from continualgp.util import vem_algorithm as onlineVEM
from continualgp.util import draw_mini_slices
from hetmogp.svmogp import SVMOGP
from hetmogp.util import vem_algorithm as VEM

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
import scipy.io as sio

# Load Data
data = sio.loadmat('../data/nasa.mat')
Y_nasa = data['nasa'][:,2]
Y_nasa = np.log(Y_nasa + 1)
Y_nasa = Y_nasa[:,np.newaxis]
X_nasa = np.linspace(0,100, Y_nasa.shape[0])[:,np.newaxis]

# Normalization
Y_nasa[:,0] = Y_nasa[:,0] - Y_nasa[:,0].mean()
Y_nasa = Y_nasa[:,0,np.newaxis]
mean_nasa = Y_nasa[:,0].mean()

N = Y_nasa.shape[0]
M = 15 # initial
Q = 1
T = N
dimX = 1
max_X = 100.0
VEM_its = 1
max_iter = 100

# Likelihood Definition
likelihoods_list = [Gaussian(sigma=1.0)]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
D = likelihood.num_output_functions(Y_metadata)
true_W_list = [np.array(([[1.0]]))]

# Streaming data generator:
def data_streaming(Y_output, X_input, T_sections):
    streaming_Y = []
    streaming_X = []
    N = Y_output.shape[0]
    slice = np.floor(N/T_sections)
    for t in range(T_sections):
        streaming_X.append(X_input[np.r_[int(t*slice):int((t+1)*slice)],:])
        streaming_Y.append(Y_output[np.r_[int(t*slice):int((t+1)*slice)], :])
    return streaming_Y, streaming_X

# Warm init / pre-training
stream_Y_init, stream_X_init = data_streaming(Y_nasa, X_nasa, 10)

stream_X_train = stream_X_init[1]
stream_Y_train = stream_Y_init[1]

for t_stack in range(2,10):
    stream_X_train = np.vstack((stream_X_train, stream_X_init[t_stack]))
    stream_Y_train = np.vstack((stream_Y_train, stream_Y_init[t_stack]))

stream_Y_cont, stream_X_cont, = data_streaming(stream_Y_train, stream_X_train, stream_Y_train.shape[0])

model_list = []
Z_list = []
q_mean_list = []

# First batch - t=1 -------------------------------
# Kernels
ls_q = np.array(([.5] * Q))
var_q = np.array(([2.] * Q))
kern_list = util.latent_functions_prior(Q, variance=var_q, lenghtscale=ls_q, input_dim=dimX, kname='rbf')

# Inducing points
Z = np.linspace(0, np.max(stream_X_init[0]), M) + 0.005*np.random.randn()
Z = Z[:, np.newaxis]

print("[Stream 1] ---------------------------------")
hetmogp_model = SVMOGP(X=[stream_X_init[0]], Y=[stream_Y_init[0]], Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)

hetmogp_model = VEM(hetmogp_model, stochastic=False, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False)
last_model = hetmogp_model
model_list.append(last_model)
Z_list.append(Z)
q_mean_list.append(hetmogp_model.q_u_means)

prev_var = hetmogp_model.kern.variance.copy() + 0.0
prev_ls = hetmogp_model.kern.lengthscale.copy() + 0.0

# Plotting
plt.figure(figsize=[12, 4])
# Continual training - t=2:T ----------------------
for t in range(1,1001):

    # # INDUCING POINTS
    if t % 25 == 0:
        M = M + 1

    iteration = 0
    marginal = np.infty

    # KERNELS
    ls_q = np.array(([.0001] * Q))
    var_q = np.array(([0.1] * Q))

    prev_ls_q = np.array(([prev_ls] * Q))
    prev_var_q = np.array(([prev_var] * Q))
    kern_list = util.latent_functions_prior(Q, variance=prev_var_q, lenghtscale=prev_ls, input_dim=1, kname='rbf')
    kern_list_old = util.latent_functions_prior(Q, variance=prev_var_q,  lenghtscale=prev_ls, input_dim=1, kname='rbf')

    Z = np.linspace(0, np.max(stream_X_cont[t]), M) + 0.005*np.random.randn()
    Z = Z[:, np.newaxis]

    print("[Stream "+str(t+1)+"] ---------------------------------")
    # Model Construction + hyperparameter setup
    online_model = ContinualGP(X=[stream_X_cont[t]], Y=[stream_Y_cont[t]], Z=Z, kern_list=kern_list, kern_list_old=kern_list_old, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)
    util.hyperparams_new_to_old(online_model, last_model)
    online_model.phi_means, online_model.phi_chols = util.variational_new_to_old_offline(last_model.q_u_means, last_model.q_u_chols)

    online_model = onlineVEM(online_model, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False, maxIter_perVEM=max_iter)

    last_model = online_model
    prev_marginal = online_model.log_likelihood().flatten()
    prev_var = online_model.kern.variance.copy() + 0.0
    model_list.append(last_model)
    Z_list.append(Z)
    q_mean_list.append(online_model.q_u_means)

    if t % 1 == 0:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ion()
        plt.show()

        # Signals
        plt.plot(stream_X_init[0], stream_Y_init[0], '-m', alpha=0.5)
        plt.plot(stream_X_cont[t], stream_Y_cont[t], 'bx')
        plt.plot(stream_X_train[0:t+1,:], stream_Y_train[0:t+1,:], 'b-', alpha=0.5)

        # Predictive
        lik_noise = 1.0**2
        pred_X = np.linspace(0.0, stream_X_cont[t][:,0]+1.5, 500)

        m_pred_new, v_pred_new = model_list[t].predictive_new(pred_X, output_function_ind=0)
        m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new)
        m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new)

        plt.plot(pred_X, m_pred_new, '-k', linewidth=2.0)
        plt.plot(pred_X, m_pred_gp_upper_new, '-k', linewidth=0.5)
        plt.plot(pred_X, m_pred_gp_lower_new, '-k', linewidth=0.5)

        plt.plot(Z[:, 0], online_model.q_u_means, '*k')
        plt.title(r'Solar Physics Data (t='+str(t)+')')
        plt.ylabel(r'Log-Average Sunspots')
        plt.xlabel(r'Time')
        plt.ylim(-4.5, 2.5)
        plt.xlim(0.0, stream_X_cont[t][:, 0] + 1.5)
        plt.draw()
        plt.pause(0.001)
        plt.clf()


