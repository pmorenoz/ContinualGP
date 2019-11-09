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
from hetmogp.svmogp import SVMOGP
from hetmogp.util import vem_algorithm as VEM

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

N = 2000 # number of samples
M = 3  # number of inducing points
Q = 1  # number of latent functions
T = 10 # number of streaming batches
max_X = 2.0 # max of X range
VEM_its = 2
max_iter = 100

# Likelihood Definition
likelihoods_list = [Gaussian(sigma=1.5)]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
D = likelihood.num_output_functions(Y_metadata)

X = np.sort(max_X*np.random.rand(N))[:, None]

# True F function
def true_f(X_input):
    f = np.empty((X_input.shape[0], 1))
    f[:,0,None] = 4.5 * np.cos(2 * np.pi * X_input + 1.5*np.pi) - \
                3 * np.sin(4.3 * np.pi * X_input + 0.3 * np.pi) + \
                5 * np.cos(7 * np.pi * X_input + 2.4 * np.pi)
    return [f]

f_train = true_f(X)

# Generation of training data Y (sampling from the likelihood dist.)
Y = likelihood.samples(F=f_train, Y_metadata=Y_metadata)[0]

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

# Streaming simulation data
stream_Y, stream_X = data_streaming(Y, X, T)

# Train/test splitting
stream_Y_train = []
stream_Y_test = []
stream_X_train = []
stream_X_test = []
for t_list in range(T):
    x_train, x_test, y_train, y_test = train_test_split(stream_X[t_list], stream_Y[t_list], test_size = 0.33, random_state = 42)
    stream_Y_train.append(y_train)
    stream_Y_test.append(y_test)
    stream_X_train.append(x_train)
    stream_X_test.append(x_test)

model_list = []
Z_list = []
q_mean_list = []

# NLPD Test Metrics
iterations = 1
n_samples = 1000
NLPD_results = np.zeros((T,T,iterations))
for t_it in range(iterations):

    # First batch - t=1 -------------------------------
    # Kernels
    ls_q = np.array(([.05] * Q))
    var_q = np.array(([.5] * Q))
    kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=1)

    # Inducing points
    Z = np.linspace(0, np.max(stream_X[0]), M) + 0.005*np.random.randn()
    Z = Z[:, np.newaxis]

    print("[Stream 1] ---------------------------------")
    hetmogp_model = SVMOGP(X=[stream_X[0]], Y=[stream_Y[0]], Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)
    Identity = np.eye(Q)

    hetmogp_model = VEM(hetmogp_model, stochastic=False, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False)
    last_model = hetmogp_model
    model_list.append(last_model)
    Z_list.append(Z)
    q_mean_list.append(hetmogp_model.q_u_means)

    prev_var = hetmogp_model.kern.variance.copy() + 0.0
    prev_ls = hetmogp_model.kern.lengthscale.copy() + 0.0

    prev_marginal = - np.infty
    # Continual training - t=2:T ----------------------
    for t in range(1,T):
        marginal = 1.0
        while marginal > 0.0 or marginal < 2*prev_marginal:
            # KERNELS
            ls_q = np.array(([.01] * Q))
            var_q = np.array(([0.5] * Q))
            prev_ls_q = np.array(([prev_ls] * Q))
            prev_var_q = np.array(([prev_var] * Q))
            kern_list = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=1)
            kern_list_old = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=1)

            # # INDUCING POINTS
            Z = np.linspace(0, np.max(stream_X[t]), (t+1)*M) + 0.005*np.random.randn()
            Z = Z[:, np.newaxis]

            print("[Stream "+str(t+1)+"] ---------------------------------")
            # Model Construction + hyperparameter setup
            online_model = ContinualGP(X=[stream_X[t]], Y=[stream_Y[t]], Z=Z, kern_list=kern_list, kern_list_old=kern_list_old, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)
            util.hyperparams_new_to_old(online_model, last_model)
            online_model.phi_means, online_model.phi_chols = util.variational_new_to_old_offline(last_model.q_u_means, last_model.q_u_chols)

            online_model = onlineVEM(online_model, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False, maxIter_perVEM=max_iter)
            marginal = online_model.log_likelihood().flatten()

        last_model = online_model
        prev_marginal = online_model.log_likelihood().flatten()
        prev_var = online_model.kern.variance.copy() + 0.0
        prev_ls = online_model.kern.lengthscale.copy() + 0.0
        model_list.append(last_model)
        Z_list.append(Z)
        q_mean_list.append(online_model.q_u_means)

    # Latex Plots
    util.plot_streaming_latex(model_list, stream_X_train, stream_X_test, stream_Y_train, stream_Y_test, Z_list, q_mean_list, save=False)

    for t_step in range(T):
        if t_step > 0:
            for t_past in range(t_step+1):
                m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_past], output_function_ind=0)
                nlogpred = - Gaussian(sigma=1.5).log_predictive(stream_Y_test[t_past], m_pred, v_pred, n_samples)
                print('Negative Log-Predictive / model t=' +str(t_step+1) + '/ batch t='+str(t_past+1)+'): ' + str(nlogpred))
                NLPD_results[t_step,t_past,t_it] = nlogpred

        else:
            m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_step], output_function_ind=0)
            nlogpred = - Gaussian(sigma=1.5).log_predictive(stream_Y_test[t_step], m_pred, v_pred, n_samples)
            print('Negative Log-Predictive / model t=' + str(t_step+1) + '/ batch t=' + str(t_step+1) + '): ' + str(nlogpred))
            NLPD_results[t_step, t_step, t_it] = nlogpred

        print('-------------------------------------------------------------------------')


