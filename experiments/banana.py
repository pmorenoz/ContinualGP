# Copyright (c) 2019 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import warnings
import os

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
from sklearn.model_selection import train_test_split

from likelihoods.bernoulli import Bernoulli

from continualgp.het_likelihood import HetLikelihood
from continualgp.continualgp import ContinualGP
from continualgp import util
from continualgp.util import vem_algorithm as onlineVEM
from hetmogp.svmogp import SVMOGP
from hetmogp.util import vem_algorithm as VEM

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
import scipy.io as sio

# Load Data
data = sio.loadmat('../data/banana.mat')
X_banana = data['banana_X']
Y_banana = data['banana_Y']

# Sorting wrt first input dimension
Y_banana = Y_banana[X_banana[:,0].argsort()]
X_banana = X_banana[X_banana[:,0].argsort()]

# plot limits
max_X = X_banana[:,0].max()
max_Y = X_banana[:,1].max()
min_X = X_banana[:,0].min()
min_Y = X_banana[:,1].min()

N, dimX = X_banana.shape
M = 3 # number of inducing points
Q = 1 # num of latent functions
T = 4 # num of streaming batches
VEM_its = 4
max_iter = 100

# Likelihood Definition
likelihoods_list = [Bernoulli()]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
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

stream_Y, stream_X = data_streaming(Y_banana, X_banana, T)

# Train/test splitting
stream_Y_train = []
stream_Y_test = []
stream_X_train = []
stream_X_test = []
for t_list in range(T):
    x_train, x_test, y_train, y_test = train_test_split(stream_X[t_list], stream_Y[t_list], test_size = 0.3, random_state = 42)
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
ME_results = np.zeros((T,T,iterations))
for t_it in range(iterations):

    # First batch - t=1 -------------------------------
    # Kernels
    ls_q = np.array(([.05] * Q))
    var_q = np.array(([.5] * Q))
    kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=dimX)

    # INDUCING POINTS
    stack_X_train = stream_X_train[0]
    stack_Y_train = stream_Y_train[0]

    mx = stack_X_train[:,0].mean()
    my = stack_X_train[:,1].mean()
    vx = stack_X_train[:,0].var()
    vy = stack_X_train[:,1].var()

    zy = np.linspace(my - 2*vy, my + 2*vy, M)
    zx = np.linspace(mx - 2*vx, mx + 2*vx, M)
    ZX, ZY = np.meshgrid(zx, zy)
    ZX = ZX.reshape(M**2,1)
    ZY = ZY.reshape(M**2,1)
    Z = np.hstack((ZX,ZY))

    print("[Stream 1] ---------------------------------")
    hetmogp_model = SVMOGP(X=[stream_X_train[0]], Y=[stream_Y_train[0]], Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)

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
        # KERNELS
        ls_q = np.array(([.05] * Q))
        var_q = np.array(([0.1] * Q))
        prev_ls_q = np.array(([prev_ls] * Q)) + np.random.rand() - np.random.rand()
        prev_var_q = np.array(([prev_var] * Q)) + np.random.rand() - np.random.rand()
        kern_list = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=2)
        kern_list_old = util.latent_functions_prior(Q, lenghtscale=prev_ls_q, variance=prev_var_q, input_dim=2)

        # INDUCING POINTS
        M = M+1
        stack_X_train = np.vstack((stack_X_train, stream_X_train[t]))
        stack_Y_train = np.vstack((stack_Y_train, stream_Y_train[t]))

        mx = stack_X_train[:, 0].mean()
        my = stack_X_train[:, 1].mean()
        vx = stack_X_train[:, 0].var()
        vy = stack_X_train[:, 1].var()

        zy = np.linspace(my - 1.5*vy, my + 1.5*vy, M)
        zx = np.linspace(mx - 2 * vx, mx + 2 * vx, M)
        ZX, ZY = np.meshgrid(zx, zy)
        ZX = ZX.reshape(M ** 2, 1)
        ZY = ZY.reshape(M ** 2, 1)
        Z = np.hstack((ZX, ZY))

        print("[Stream "+str(t+1)+"] ---------------------------------")
        # Model Construction + hyperparameter setup
        online_model = ContinualGP(X=[stream_X_train[t]], Y=[stream_Y_train[t]], Z=Z, kern_list=kern_list, kern_list_old=kern_list_old, likelihood=likelihood, Y_metadata=Y_metadata, W_list=true_W_list)
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

    max_min = [min_X, min_Y, max_X, max_Y]
    # plotting takes a bit of time (~ 30s-1min)
    util.plot_banana_latex(model_list, stream_X_train, stream_X_test, stream_Y_train, stream_Y_test, Z_list, max_min, save=False)

    for t_step in range(T):
        if t_step > 0:
            for t_past in range(t_step + 1):
                m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_past], output_function_ind=0)
                nlogpred = - Bernoulli().log_predictive(stream_Y_test[t_past], m_pred, v_pred, n_samples)
                m_pred = np.exp(m_pred) / (1 + np.exp(m_pred))
                m_pred[m_pred[:, 0] < 0.5, 0] = 0.0
                m_pred[m_pred[:, 0] >= 0.5, 0] = 1.0

                errors = stream_Y_test[t_past] - m_pred
                errors = np.abs(errors)
                me = np.sum(errors, axis=0) / errors.shape[0]

                print('Negative Log-Predictive / model t=' + str(t_step + 1) + '/ batch t=' + str(t_past + 1) + '): ' + str(nlogpred))
                print('Mean Error/ model t=' + str(t_step + 1) + '/ batch t=' + str(t_past + 1) + '): ' + str(me))
                NLPD_results[t_step, t_past, t_it] = nlogpred
                ME_results[t_step, t_past, t_it] = me


        else:
            m_pred, v_pred = model_list[t_step].predictive_new(stream_X_test[t_step], output_function_ind=0)
            nlogpred = - Bernoulli().log_predictive(stream_Y_test[t_step], m_pred, v_pred, n_samples)
            m_pred = np.exp(m_pred) / (1 + np.exp(m_pred))
            m_pred[m_pred[:, 0] < 0.5, 0] = 0.0
            m_pred[m_pred[:, 0] >= 0.5, 0] = 1.0

            errors = stream_Y_test[t_step] - m_pred
            errors = np.abs(errors)
            me = np.sum(errors, axis=0) / errors.shape[0]

            print('Negative Log-Predictive / model t=' + str(t_step + 1) + '/ batch t=' + str(t_step + 1) + '): ' + str(nlogpred))
            print('Mean Error / model t=' + str(t_step + 1) + '/ batch t=' + str(t_step + 1) + '): ' + str(me))
            NLPD_results[t_step, t_step, t_it] = nlogpred
            ME_results[t_step, t_step, t_it] = me

        print('-------------------------------------------------------------------------')