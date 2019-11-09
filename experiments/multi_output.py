# Copyright (c) 2019 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import warnings
import os

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
from sklearn.model_selection import train_test_split

from likelihoods.gaussian import Gaussian

from continualgp.het_likelihood import HetLikelihood
from continualgp.continualgp import ContinualGP
from continualgp import util
from continualgp.util import vem_algorithm as onlineVEM
from continualgp.util import draw_mini_slices
from hetmogp.svmogp import SVMOGP
from hetmogp.util import vem_algorithm as VEM

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


N = 2000 # number of samples
M = 5  # number of inducing points
Q = 2  # number of latent functions
T = 5 # number of streaming batches
max_X = 2.0 # max of X range
VEM_its = 2

# Heterogeneous Likelihood Definition
likelihoods_list = [Gaussian(sigma=1.), Gaussian(sigma=2.0)]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
D = likelihood.num_output_functions(Y_metadata)

X1 = np.sort(max_X * np.random.rand(N))[:, None]
X2 = np.sort(max_X * np.random.rand(N))[:, None]
X = [X1, X2]

# True U functions
def true_u_functions(X_list):
    u_functions = []
    for X in X_list:
        u_task = np.empty((X.shape[0], 2))
        u_task[:, 0, None] = 4.5 * np.cos(2 * np.pi * X + 1.5 * np.pi) - \
                             3 * np.sin(4.3 * np.pi * X + 0.3 * np.pi) + \
                             5 * np.cos(7 * np.pi * X + 2.4 * np.pi)
        u_task[:, 1, None] = 4.5 * np.cos(1.5 * np.pi * X + 0.5 * np.pi) + \
                             5 * np.sin(3 * np.pi * X + 1.5 * np.pi) - \
                             5.5 * np.cos(8 * np.pi * X + 0.25 * np.pi)

        u_functions.append(u_task)
    return u_functions

# True F functions
def true_f_functions(true_u, X_list):
    true_f = []
    W = W_lincombination()
    # D=1
    for d in range(2):
        f_d = np.zeros((X_list[d].shape[0], 1))
        for q in range(2):
            f_d += W[q][d].T * true_u[d][:, q, None]
        true_f.append(f_d)

    return true_f, W

# True W combinations
def W_lincombination():
    W_list = []
    # q=1
    Wq1 = np.array(([[-0.5], [0.1]]))
    W_list.append(Wq1)
    # q=2
    Wq2 = np.array(([[-0.1], [.6]]))
    W_list.append(Wq2)
    return W_list

# True functions values for inputs X
trueU = true_u_functions(X)
trueF, trueW_list = true_f_functions(trueU, X)

# Generating training data Y (sampling from heterogeneous likelihood)
Y = likelihood.samples(F=trueF, Y_metadata=Y_metadata)

# Streaming data generator
def data_streaming(Y_output, X_input, T_sections):
    streaming_Y = []
    streaming_X = []
    N = Y_output[0].shape[0]
    slice = np.floor(N / T_sections)
    num_outputs = len(Y_output)
    for t in range(T_sections):
        stream_X_d = []
        stream_Y_d = []
        for d in range(num_outputs):
            stream_X_d.append(X_input[d][np.r_[int(t*slice):int((t+1)*slice)],:])
            stream_Y_d.append(Y_output[d][np.r_[int(t*slice):int((t+1)*slice)], :])

        streaming_X.append(stream_X_d)
        streaming_Y.append(stream_Y_d)

    return streaming_Y, streaming_X

# Streaming simulation data
stream_Y, stream_X = data_streaming(Y, X, T)

# Train/test splitting
stream_Y_train = []
stream_Y_test = []
stream_X_train = []
stream_X_test = []
num_outputs = len(stream_Y[0])
for t_list in range(T):
    sY_train_d = []
    sY_test_d = []
    sX_train_d = []
    sX_test_d = []
    for d in range(num_outputs):
        x_train, x_test, y_train, y_test = train_test_split(stream_X[t_list][d], stream_Y[t_list][d], test_size=0.33,random_state=42)
        sY_train_d.append(y_train)
        sY_test_d.append(y_test)
        sX_train_d.append(x_train)
        sX_test_d.append(x_test)

    stream_Y_train.append(sY_train_d)
    stream_Y_test.append(sY_test_d)
    stream_X_train.append(sX_train_d)
    stream_X_test.append(sX_test_d)

model_list = []
Z_list = []
q_mean_list = []

# NLPD Test Metrics
iterations = 1
n_samples = 1000
NLPD_results = [np.zeros((T,T,iterations)), np.zeros((T,T,iterations))]
for t_it in range(iterations):

    print('Iteration: '+str(t_it))

    # KERNELS
    ls_q = np.array(([.05] * Q))
    var_q = np.array(([.5] * Q))
    kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=1)

    # # INDUCING POINTS
    Z = np.linspace(0, np.max(stream_X[0]), M)
    Z = Z[:, np.newaxis]

    # t=1 ---------------------- First Stream: D_1 --------------------
    # HetMOGP Model
    print("[Stream 1] ---------------------------------")
    hetmogp_model = SVMOGP(X=stream_X[0], Y=stream_Y[0], Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata, W_list=trueW_list)
    hetmogp_model = VEM(hetmogp_model, stochastic=False, vem_iters=3, optZ=False, verbose=False, verbose_plot=False, non_chained=False)

    last_model = hetmogp_model
    model_list.append(last_model)
    Z_list.append(Z)
    q_mean_list.append(hetmogp_model.q_u_means)

    prev_var = hetmogp_model.kern.variance.copy() + 0.0
    prev_ls = hetmogp_model.kern.lengthscale.copy() + 0.0

    prev_marginal = - np.infty
    for t in range(1,T):
        marginal = 1.0
        while marginal > 0.0 or marginal < 2 * prev_marginal:
            # Kernels
            ls_q = np.array(([.01] * Q))
            var_q = np.array(([0.5] * Q))
            prev_ls_q = np.array(([prev_ls] * Q))
            prev_var_q = np.array(([prev_var] * Q))
            kern_list = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=1)
            kern_list_old = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=1)

            # Inducing Points
            Z = np.linspace(0, np.max(stream_X[t]), (t+1)*M) + 0.005*np.random.randn()
            Z = Z[:, np.newaxis]

            print("[Stream " + str(t + 1) + "] ---------------------------------")
            online_model = ContinualGP(X=stream_X[t], Y=stream_Y[t], Z=Z, kern_list=kern_list, kern_list_old=kern_list_old, likelihood=likelihood, Y_metadata=Y_metadata, W_list=trueW_list)
            util.hyperparams_new_to_old(online_model, last_model)
            online_model.phi_means, online_model.phi_chols = util.variational_new_to_old_offline(last_model.q_u_means, last_model.q_u_chols)
            online_model = onlineVEM(online_model, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False)
            marginal = online_model.log_likelihood().flatten()

        last_model = online_model
        prev_marginal = online_model.log_likelihood().flatten()
        prev_var = online_model.kern.variance.copy() + 0.0
        prev_ls = online_model.kern.lengthscale.copy() + 0.0
        model_list.append(last_model)
        Z_list.append(Z)
        q_mean_list.append(online_model.q_u_means)


    # Latex Plots
    util.plot_multioutput_latex(model_list, stream_X_train, stream_X_test, stream_Y_train, stream_Y_test, Z_list, q_mean_list, save=False)

    for t_step in range(T):
        for d in range(D):
            if t_step > 0:
                for t_past in range(t_step+1):
                    m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_past][d], output_function_ind=d)
                    nlogpred = - model_list[t_step].likelihood.likelihoods_list[d].log_predictive(stream_Y_test[t_past][d], m_pred, v_pred, n_samples)
                    print('Negative Log-Predictive / model t=' +str(t_step+1) + '/ batch t='+str(t_past+1)+'): ' + str(nlogpred))
                    NLPD_results[d][t_step,t_past,t_it] = nlogpred

            else:
                m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_step][d], output_function_ind=d)
                nlogpred = - model_list[t_step].likelihood.likelihoods_list[d].log_predictive(stream_Y_test[t_step][d], m_pred, v_pred, n_samples)
                print('Negative Log-Predictive / model t=' + str(t_step+1) + '/ batch t=' + str(t_step+1) + '): ' + str(nlogpred))
                NLPD_results[d][t_step, t_step, t_it] = nlogpred

            print('-------------------------------------------------------------------------')
