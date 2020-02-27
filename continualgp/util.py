# Copyright (c) 2019 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield


from GPy import kern
from GPy.util import choleskies
from GPy.util import linalg
from GPy.core.parameterization.param import Param
import random
import warnings
import numpy as np
import climin
from functools import partial
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save


def  get_batch_scales(X_all, X):
    batch_scales = []
    for t, X_all_task in enumerate(X_all):
        batch_scales.append(float(X_all_task.shape[0]) / float(X[t].shape[0]))
    return batch_scales

def true_u_functions(X_list, Q):
    u_functions = []
    amplitude = (1.5-0.5)*np.random.rand(Q,3) + 0.5
    freq = (3-1)*np.random.rand(Q,3) + 1
    shift = 2*np.random.rand(Q,3)
    for X in X_list:
        u_task = np.empty((X.shape[0],Q))
        for q in range(Q):
            u_task[:,q,None] = 3*amplitude[q,0]*np.cos(freq[q,0]*np.pi*X + shift[q,0]*np.pi) - \
                               2*amplitude[q,1]*np.sin(2*freq[q,1]*np.pi*X + shift[q,1]*np.pi) + \
                               amplitude[q,2] * np.cos(4*freq[q, 2] * np.pi * X + shift[q, 2] * np.pi)

        u_functions.append(u_task)
    return u_functions

def true_f_functions(true_u, W_list, D, likelihood_list, Y_metadata):
    true_f = []
    f_index = Y_metadata['function_index'].flatten()
    d_index = Y_metadata['d_index'].flatten()
    for t, u_task in enumerate(true_u):
        Ntask = u_task.shape[0]
        _, num_f_task, _ = likelihood_list[t].get_metadata()
        F = np.zeros((Ntask, num_f_task))
        for q, W in enumerate(W_list):
            for d in range(D):
                if f_index[d] == t:
                    F[:,d_index[d],None] += np.tile(W[d].T, (Ntask, 1)) * u_task[:, q, None]

        true_f.append(F)
    return true_f

def mini_slices(n_samples, batch_size):
    """Yield slices of size `batch_size` that work with a container of length
    `n_samples`."""
    n_batches, rest = divmod(n_samples, batch_size)
    if rest != 0:
        n_batches += 1

    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def draw_mini_slices(n_samples, batch_size, with_replacement=False):
    slices = mini_slices(n_samples, batch_size)
    idxs = list(range(len(slices)))  # change this line

    if with_replacement:
        yield random.choice(slices)
    else:
        while True:
            random.shuffle(list(idxs))
            for i in idxs:
                yield slices[i]


def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None, kname=None):
    if lenghtscale is None:
        lenghtscale = np.random.rand(Q)
    else:
        lenghtscale = lenghtscale

    if variance is None:
        variance = np.random.rand(Q)
    else:
        variance = variance

    if kname is None:
        kname = 'rbf'
    else:
        kname = kname

    kern_list = []
    for q in range(Q):
        if kname=='rbf':
            kern_q = kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
        else:
            kern_q = kern.Matern32(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='matern')

        kern_q.name = 'kern_q'+str(q)
        kern_list.append(kern_q)
    return kern_list

def random_W_kappas(Q,D,rank, experiment=False):
    W_list = []
    kappa_list = []
    for q in range(Q):
        p = np.random.binomial(n=1, p=0.5*np.ones((D,1)))
        Ws = p*np.random.normal(loc=0.5, scale=0.5, size=(D,1)) - (p-1)*np.random.normal(loc=-0.5, scale=0.5, size=(D,1))
        W_list.append(Ws / np.sqrt(rank)) # deberían ser tanto positivos como negativos
        if experiment:
            kappa_list.append(np.zeros(D))
        else:
            kappa_list.append(np.zeros(D))
    return W_list, kappa_list


def ICM(input_dim, output_dim, kernel, rank, W=None, kappa=None, name='ICM'):
    """
    Builds a kernel for an Intrinsic Coregionalization Model
    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    kern_q = kernel.copy()
    if kernel.input_dim != input_dim:
        kernel.input_dim = input_dim
        warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")
    B = kern.Coregionalize(input_dim=input_dim, output_dim=output_dim, rank=rank, W=W, kappa=kappa)
    B.name = name
    K = kern_q.prod(B, name=name)
    return K, B


def LCM(input_dim, output_dim, kernels_list, W_list, kappa_list, rank, name='B_q'):
    """
    Builds a kernel for an Linear Coregionalization Model
    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    B_q = []
    K, B = ICM(input_dim, output_dim, kernels_list[0], W=W_list[0], kappa=kappa_list[0], rank=rank, name='%s%s' %(name,0))
    B_q.append(B)
    for q, kernel in enumerate(kernels_list[1:]):
        Kq, Bq = ICM(input_dim, output_dim, kernel, W=W_list[q+1], kappa=kappa_list[q+1], rank=rank, name='%s%s' %(name,q+1))
        B_q.append(Bq)
        K += Kq
    return K, B_q

def cross_covariance(X, Z, B, kernel_list, d):
    """
    Builds the cross-covariance cov[f_d(x),u(z)] of a Multi-output GP
    :param X: Input data
    :param Z: Inducing Points
    :param B: Coregionalization matric
    :param kernel_list: Kernels of u_q functions
    :param d: output function f_d index
    :return: Kfdu
    """
    N,_ = X.shape
    M,Dz = Z.shape
    Q = len(B)
    Xdim = int(Dz/Q)
    Kfdu = np.empty([N,M*Q])
    for q, B_q in enumerate(B):
        Kfdu[:, q * M:(q * M) + M] = B_q.W[d] * kernel_list[q].K(X, Z[:, q*Xdim:q*Xdim+Xdim])
        #Kfdu[:,q*M:(q*M)+M] = B_q.W[d]*kernel_list[q].K(X,Z[:,q,None])
        #Kfdu[:, q * M:(q * M) + M] = B_q.B[d,d] * kernel_list[q].K(X, Z[:,q,None])
    return Kfdu

def function_covariance(X, B, kernel_list, d):
    """
    Builds the cross-covariance Kfdfd = cov[f_d(x),f_d(x)] of a Multi-output GP
    :param X: Input data
    :param B: Coregionalization matrix
    :param kernel_list: Kernels of u_q functions
    :param d: output function f_d index
    :return: Kfdfd
    """
    N,_ = X.shape
    Kfdfd = np.zeros((N, N))
    for q, B_q in enumerate(B):
        Kfdfd += B_q.B[d,d]*kernel_list[q].K(X,X)
    return Kfdfd

def latent_funs_cov(Z, kernel_list):
    """
    Builds the full-covariance cov[u(z),u(z)] of a Multi-output GP
    for a Sparse approximation
    :param Z: Inducing Points
    :param kernel_list: Kernels of u_q functions priors
    :return: Kuu
    """
    Q = len(kernel_list)
    M,Dz = Z.shape
    Xdim = int(Dz/Q)
    #Kuu = np.zeros([Q*M,Q*M])
    Kuu = np.empty((Q, M, M))
    Luu = np.empty((Q, M, M))
    Kuui = np.empty((Q, M, M))
    for q, kern in enumerate(kernel_list):
        Kuu[q, :, :] = kern.K(Z[:,q*Xdim:q*Xdim+Xdim],Z[:,q*Xdim:q*Xdim+Xdim])
        Luu[q, :, :] = linalg.jitchol(Kuu[q, :, :])
        Kuui[q, :, :], _ = linalg.dpotri(np.asfortranarray(Luu[q, :, :]))
    return Kuu, Luu, Kuui

def latent_funs_conditional(Z, Zold, kernel_list):
    Q = len(kernel_list)
    M, Dz = Z.shape
    Mold, Dz_old = Zold.shape
    Xdim = int(Dz/Q)
    Kuu_cond = np.empty((Q, M, Mold))
    for q, kern in enumerate(kernel_list):
        Kuu_cond[q, :, :] = kern.K(Z[:,q*Xdim:q*Xdim+Xdim], Zold[:,q*Xdim:q*Xdim+Xdim])
    return Kuu_cond

def conditional_prior(Z, Zold, kern_list_old, phi_means, phi_chols):
    M, Dz = Z.shape
    Mold, _ = Zold.shape
    Q = len(kern_list_old)

    # Algebra for q(u):
    #phi_m = phi_means.copy()
    phi_L = choleskies.flat_to_triang(phi_chols)
    phi_S = np.empty((Q, Mold, Mold))
    [np.dot(phi_L[q, :, :], phi_L[q, :, :].T, phi_S[q, :, :]) for q in range(Q)]

    Mu = np.empty((Q, M, 1))
    Kuu = np.empty((Q, M, M))
    Luu = np.empty((Q, M, M))
    Kuui = np.empty((Q, M, M))

    Kuu_old, Luu_old, Kuui_old = latent_funs_cov(Zold, kern_list_old)
    Kuu_new, _, _ = latent_funs_cov(Z, kern_list_old)
    Kuu_cond = latent_funs_conditional(Z, Zold, kern_list_old)

    for q, kern in enumerate(kern_list_old):
        R, _ = linalg.dpotrs(np.asfortranarray(Luu_old[q, :, :]), Kuu_cond[q, :, :].T)
        Auu = R.T # Kuu_cond * Kuui
        Mu[q, :] = np.dot(Auu, phi_means[:, q, None])
        Kuu[q, :, :] = Kuu_new[q, :, :] + np.dot(np.dot(R.T, phi_S[q, :, :]), R) - np.dot(Kuu_cond[q, :, :], R)
        Luu[q, :, :] = linalg.jitchol(Kuu[q, :, :])
        Kuui[q, :, :], _ = linalg.dpotri(np.asfortranarray(Luu[q, :, :]))
    return Mu, Kuu, Luu, Kuui

def hyperparams_new_to_old(model_new, model_old):
    model_new.update_model(False)
    Q = len(model_old.kern_list)
    model_new.Zold = model_old.Z.copy() + 0.
    for q in range(Q):
        model_new.kern_list_old[q].lengthscale = model_old.kern_list[q].lengthscale.copy() + 0.
        model_new.kern_list_old[q].variance = model_old.kern_list[q].variance.copy() + 0.
        #print(model_new.B_list[q].W[0,0])# = 0
        #model_new.B_list[q].W[0,0] = 0.
        #print(model_new.B_list[q].W[0, 0])
        #print(model_new.B_list[q].W.shape)
        #print(model_old.B_list[q].W.copy() + 0.)

        model_new.B_list[q].W[:] = model_old.B_list[q].W + 0.

    model_new.initialize_parameter()


def variational_new_to_old_online(q_new_means, q_new_chols):
    #q_old_means = Param('phi', q_new_means + 0.)
    #q_old_chols = Param('LO_chols', q_new_chols + 0.)
    q_old_means = Param('phi', q_new_means)
    q_old_chols = Param('LO_chols', q_new_chols)
    return q_old_means, q_old_chols

def variational_new_to_old_offline(q_old_means, q_old_chols):

    means_old = q_old_means.copy() + 0.
    chols_old = q_old_chols.copy() + 0.

    return means_old, chols_old

def generate_toy_U(X,Q):
    arg = np.tile(X, (1,Q))
    rnd = np.tile(np.random.rand(1,Q), (X.shape))
    U = 2*rnd*np.sin(10*rnd*arg + np.random.randn(1)) + 2*rnd*np.cos(20*rnd*arg + np.random.randn(1))
    return U

def _gradient_reduce_numpy(coreg, dL_dK, index, index2):
    index, index2 = index[:,0], index2[:,0]
    dL_dK_small = np.zeros_like(coreg.B)
    for i in range(coreg.output_dim):
        tmp1 = dL_dK[index==i]
        for j in range(coreg.output_dim):
            dL_dK_small[j,i] = tmp1[:,index2==j].sum()
    return dL_dK_small

def _gradient_B(coreg, dL_dK, index, index2):
    index, index2 = index[:,0], index2[:,0]
    B = coreg.B
    isqrtB = 1 / np.sqrt(B)
    dL_dK_small = np.zeros_like(B)
    for i in range(coreg.output_dim):
        tmp1 = dL_dK[index==i]
        for j in range(coreg.output_dim):
            dL_dK_small[j,i] = (0.5 * isqrtB[i,j] * tmp1[:,index2==j]).sum()
    return dL_dK_small

def update_gradients_diag(coreg, dL_dKdiag):
    dL_dKdiag_small = np.array([dL_dKdiag_task.sum() for dL_dKdiag_task in dL_dKdiag])
    coreg.W.gradient = 2.*coreg.W*dL_dKdiag_small[:, None] # should it be 2*..? R/Yes Pablo, it should be :)
    coreg.kappa.gradient = dL_dKdiag_small

def update_gradients_full(coreg, dL_dK, X, X2=None):
    index = np.asarray(X, dtype=np.int)
    if X2 is None:
        index2 = index
    else:
        index2 = np.asarray(X2, dtype=np.int)

    dL_dK_small = _gradient_reduce_numpy(coreg, dL_dK, index, index2)
    dkappa = np.diag(dL_dK_small).copy()
    dL_dK_small += dL_dK_small.T
    dW = (coreg.W[:, None, :]*dL_dK_small[:, :, None]).sum(0)

    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

def update_gradients_Kmn(coreg, dL_dK, D):
    dW = np.zeros((D,1))
    dkappa = np.zeros((D)) # not used
    for d in range(D):
        dW[d,:] = dL_dK[d].sum()

    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

def gradients_coreg(coreg, dL_dK, X, X2=None):
    index = np.asarray(X, dtype=np.int)
    if X2 is None:
        index2 = index
    else:
        index2 = np.asarray(X2, dtype=np.int)

    dK_dB = _gradient_B(coreg, dL_dK, index, index2)
    dkappa = np.diag(dK_dB).copy()
    dK_dB += dK_dB.T
    dW = (coreg.W[:, None, :]*dK_dB[:, :, None]).sum(0)
    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

def gradients_coreg_diag(coreg, dL_dKdiag, kern_q, X, X2=None):
    # dL_dKdiag is (NxD)
    if X2 is None:
        X2 = X
    N,D =  dL_dKdiag.shape
    matrix_sum = np.zeros((D,1))
    for d in range(D):
        matrix_sum[d,0] = np.sum(np.diag(kern_q.K(X, X2)) * dL_dKdiag[:,d,None])

    dW = 2 * coreg.W * matrix_sum
    dkappa = matrix_sum
    return dW, dkappa

def vem_algorithm(model, vem_iters=None, maxIter_perVEM = None, step_rate=None ,verbose=False, optZ=True, verbose_plot=False, non_chained=True):
    if vem_iters is None:
        vem_iters = 5
    if maxIter_perVEM is None:
        #maxIter_perVEM = 25
        maxIter_perVEM = 100

    model['.*.kappa'].fix() # must be always fixed
    #model.elbo = np.empty((vem_iters,1))

    if model.batch_size is None:

        for i in range(vem_iters):
            # VARIATIONAL E-STEP
            model['.*.lengthscale'].fix()
            model['.*.variance'].fix()
            model.Z.fix()
            model['.*.W'].fix()

            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            model.optimize(messages=verbose, max_iters=maxIter_perVEM)
            print('iteration ('+str(i+1)+') VE step, ELBO='+str(model.log_likelihood().flatten()))

            # VARIATIONAL M-STEP
            model['.*.lengthscale'].unfix()
            model['.*.variance'].unfix()
            if optZ:
                model.Z.unfix()
            if non_chained:
                model['.*.W'].unfix()

            model.q_u_means.fix()
            model.q_u_chols.fix()
            model.optimize(messages=verbose, max_iters=maxIter_perVEM)
            print('iteration (' + str(i+1) + ') VM step, ELBO=' + str(model.log_likelihood().flatten()))

    else:

        if step_rate is None:
            step_rate = 0.01

        model.elbo = np.empty((maxIter_perVEM*vem_iters+2, 1))
        model.elbo[0,0]=model.log_likelihood()
        c_full = partial(model.callback, max_iter=maxIter_perVEM, verbose=verbose, verbose_plot=verbose_plot)

        for i in range(vem_iters):
            # VARIATIONAL E-STEP
            model['.*.lengthscale'].fix()
            model['.*.variance'].fix()
            model.Z.fix()
            model['.*.W'].fix()

            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            optimizer = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate,decay_mom1=1 - 0.9, decay_mom2=1 - 0.999)
            optimizer.minimize_until(c_full)
            print('iteration (' + str(i + 1) + ') VE step, mini-batch ELBO=' + str(model.log_likelihood().flatten()))
            #
            # # VARIATIONAL M-STEP
            model['.*.lengthscale'].unfix()
            model['.*.variance'].unfix()
            if optZ:
                model.Z.unfix()
            if non_chained:
                model['.*.W'].unfix()

            model.q_u_means.fix()
            model.q_u_chols.fix()
            optimizer = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate,decay_mom1=1 - 0.9, decay_mom2=1 - 0.999)
            optimizer.minimize_until(c_full)
            print('iteration (' + str(i + 1) + ') VM step, mini-batch ELBO=' + str(model.log_likelihood().flatten()))

    # Unfix everything
    model.q_u_means.unfix()
    model.q_u_chols.unfix()
    model['.*.lengthscale'].unfix()
    model['.*.variance'].unfix()
    model.Z.unfix()
    model['.*.W'].unfix()

    return model

def plot_streaming_figures_experiment1_latex(model_list, Xtrain_list, Xtest_list, Ytrain_list, Ytest_list):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # First BATCH
    m_pred_gaussian1, v_pred_gaussian1 = model_list[0].predictive_new(Xtest_list[0][0], output_function_ind=0)
    m_pred_gp_upper_gaussian1 = m_pred_gaussian1 + 2 * np.sqrt(v_pred_gaussian1)
    m_pred_gp_lower_gaussian1 = m_pred_gaussian1 - 2 * np.sqrt(v_pred_gaussian1)

    m_pred_gaussian2, v_pred_gaussian2 = model_list[0].predictive_new(Xtest_list[0][1], output_function_ind=1)
    m_pred_gp_upper_gaussian2 = m_pred_gaussian2 + 2 * np.sqrt(v_pred_gaussian2)
    m_pred_gp_lower_gaussian2 = m_pred_gaussian2 - 2 * np.sqrt(v_pred_gaussian2)

    fig_batch_1 = plt.figure(figsize=(12, 5))
    plt.plot(Xtrain_list[0][0], Ytrain_list[0][0], 'x', color='blue', markersize=10, alpha=0.2)
    plt.plot(Xtest_list[0][0], Ytest_list[0][0], 'o', color='blue', markersize=2, alpha=0.75)
    plt.plot(Xtest_list[0][0], m_pred_gaussian1, 'b-', linewidth=4, alpha=0.5)
    plt.plot(Xtest_list[0][0], m_pred_gp_upper_gaussian1, 'b-', linewidth=2, alpha=1)
    plt.plot(Xtest_list[0][0], m_pred_gp_lower_gaussian1, 'b-', linewidth=2, alpha=1)

    plt.plot(Xtrain_list[0][1], Ytrain_list[0][1], 'x', color='red', markersize=10, alpha=0.2)
    plt.plot(Xtest_list[0][1], Ytest_list[0][1], 'o', color='red', markersize=2, alpha=0.75)
    plt.plot(Xtest_list[0][1], m_pred_gaussian2, 'r-', linewidth=4, alpha=0.5)
    plt.plot(Xtest_list[0][1], m_pred_gp_upper_gaussian2, 'r-', linewidth=2, alpha=1)
    plt.plot(Xtest_list[0][1], m_pred_gp_lower_gaussian2, 'r-', linewidth=2, alpha=1)

    for q in range(model_list[0].Z.shape[1]):
        for m in range(model_list[0].Z[:,q].shape[0]):
            plt.axvline(model_list[0].Z[m, q], color='black', alpha=0.5)

    plt.title(r'Online Multi-Output Gaussian Regression (t=1)')
    plt.ylabel(r'Real Outputs')
    plt.xlabel(r'Real Inputs')
    plt.xlim(0, 1)
    #tikz_save('online_mogp_regression_batch_1.tex')
    plt.show()

    # -------------------------------------------------------------------------------

    # Second BATCH
    joint_Xtrain_for_batch_2_output0 = np.vstack((Xtrain_list[0][0], Xtrain_list[1][0]))
    joint_Ytrain_for_batch_2_output0 = np.vstack((Ytrain_list[0][0], Ytrain_list[1][0]))
    joint_Xtest_for_batch_2_output0 = np.vstack((Xtest_list[0][0], Xtest_list[1][0]))
    joint_Ytest_for_batch_2_output0 = np.vstack((Ytest_list[0][0], Ytest_list[1][0]))

    joint_Xtrain_for_batch_2_output1 = np.vstack((Xtrain_list[0][1], Xtrain_list[1][1]))
    joint_Ytrain_for_batch_2_output1 = np.vstack((Ytrain_list[0][1], Ytrain_list[1][1]))
    joint_Xtest_for_batch_2_output1 = np.vstack((Xtest_list[0][1], Xtest_list[1][1]))
    joint_Ytest_for_batch_2_output1 = np.vstack((Ytest_list[0][1], Ytest_list[1][1]))

    m_pred2_gaussian1, v_pred2_gaussian1 = model_list[1].predictive_new(np.sort(joint_Xtest_for_batch_2_output0),
                                                                        output_function_ind=0)
    m_pred2_gp_upper_gaussian1 = m_pred2_gaussian1 + 2 * np.sqrt(v_pred2_gaussian1)
    m_pred2_gp_lower_gaussian1 = m_pred2_gaussian1 - 2 * np.sqrt(v_pred2_gaussian1)

    m_pred2_gaussian2, v_pred2_gaussian2 = model_list[1].predictive_new(np.sort(joint_Xtest_for_batch_2_output1),
                                                                        output_function_ind=1)
    m_pred2_gp_upper_gaussian2 = m_pred2_gaussian2 + 2 * np.sqrt(v_pred2_gaussian2)
    m_pred2_gp_lower_gaussian2 = m_pred2_gaussian2 - 2 * np.sqrt(v_pred2_gaussian2)

    fig_batch_2 = plt.figure(figsize=(12, 5))
    plt.plot(joint_Xtrain_for_batch_2_output0, joint_Ytrain_for_batch_2_output0, 'x', color='blue', markersize=10,
             alpha=0.2)
    plt.plot(joint_Xtest_for_batch_2_output0, joint_Ytest_for_batch_2_output0, 'o', color='blue', markersize=2,
             alpha=0.75)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output0), m_pred2_gaussian1, 'b-', linewidth=4, alpha=0.5)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output0), m_pred2_gp_upper_gaussian1, 'b-', linewidth=2, alpha=1)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output0), m_pred2_gp_lower_gaussian1, 'b-', linewidth=2, alpha=1)

    plt.plot(joint_Xtrain_for_batch_2_output1, joint_Ytrain_for_batch_2_output1, 'x', color='red', markersize=10,
             alpha=0.2)
    plt.plot(joint_Xtest_for_batch_2_output1, joint_Ytest_for_batch_2_output1, 'o', color='red', markersize=2,
             alpha=0.75)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output1), m_pred2_gaussian2, 'r-', linewidth=4, alpha=0.5)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output1), m_pred2_gp_upper_gaussian2, 'r-', linewidth=2, alpha=1)
    plt.plot(np.sort(joint_Xtest_for_batch_2_output1), m_pred2_gp_lower_gaussian2, 'r-', linewidth=2, alpha=1)

    for q in range(model_list[1].Z.shape[1]):
        for m in range(model_list[1].Z[:,q].shape[0]):
            plt.axvline(model_list[1].Z[m, q], color='black', alpha=0.5)

    plt.title(r'Online Multi-Output Gaussian Regression (t=2)')
    plt.ylabel(r'Real Outputs')
    plt.xlabel(r'Real Inputs')
    plt.xlim(0, 1)
    #tikz_save('online_mogp_regression_batch_2.tex')
    plt.show()

    # -------------------------------------------------------------------------------

    # Third BATCH
    joint_Xtrain_for_batch_3_output0 = np.vstack((joint_Xtrain_for_batch_2_output0, Xtrain_list[2][0]))
    joint_Ytrain_for_batch_3_output0 = np.vstack((joint_Ytrain_for_batch_2_output0, Ytrain_list[2][0]))
    joint_Xtest_for_batch_3_output0 = np.vstack((joint_Xtest_for_batch_2_output0, Xtest_list[2][0]))
    joint_Ytest_for_batch_3_output0 = np.vstack((joint_Ytest_for_batch_2_output0, Ytest_list[2][0]))

    joint_Xtrain_for_batch_3_output1 = np.vstack((joint_Xtrain_for_batch_2_output1, Xtrain_list[2][1]))
    joint_Ytrain_for_batch_3_output1 = np.vstack((joint_Ytrain_for_batch_2_output1, Ytrain_list[2][1]))
    joint_Xtest_for_batch_3_output1 = np.vstack((joint_Xtest_for_batch_2_output1, Xtest_list[2][1]))
    joint_Ytest_for_batch_3_output1 = np.vstack((joint_Ytest_for_batch_2_output1, Ytest_list[2][1]))

    m_pred3_gaussian1, v_pred3_gaussian1 = model_list[2].predictive_new(np.sort(joint_Xtest_for_batch_3_output0),
                                                                        output_function_ind=0)
    m_pred3_gp_upper_gaussian1 = m_pred3_gaussian1 + 2 * np.sqrt(v_pred3_gaussian1)
    m_pred3_gp_lower_gaussian1 = m_pred3_gaussian1 - 2 * np.sqrt(v_pred3_gaussian1)

    m_pred3_gaussian2, v_pred3_gaussian2 = model_list[2].predictive_new(np.sort(joint_Xtest_for_batch_3_output1),
                                                                        output_function_ind=1)
    m_pred3_gp_upper_gaussian2 = m_pred3_gaussian2 + 2 * np.sqrt(v_pred3_gaussian2)
    m_pred3_gp_lower_gaussian2 = m_pred3_gaussian2 - 2 * np.sqrt(v_pred3_gaussian2)

    fig_batch_3 = plt.figure(figsize=(12, 5))
    plt.plot(joint_Xtrain_for_batch_3_output0, joint_Ytrain_for_batch_3_output0, 'x', color='blue', markersize=10,
             alpha=0.2)
    plt.plot(joint_Xtest_for_batch_3_output0, joint_Ytest_for_batch_3_output0, 'o', color='blue', markersize=2,
             alpha=0.75)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output0), m_pred3_gaussian1, 'b-', linewidth=4, alpha=0.5)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output0), m_pred3_gp_upper_gaussian1, 'b-', linewidth=2, alpha=1)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output0), m_pred3_gp_lower_gaussian1, 'b-', linewidth=2, alpha=1)

    plt.plot(joint_Xtrain_for_batch_3_output1, joint_Ytrain_for_batch_3_output1, 'x', color='red', markersize=10,
             alpha=0.2)
    plt.plot(joint_Xtest_for_batch_3_output1, joint_Ytest_for_batch_3_output1, 'o', color='red', markersize=2,
             alpha=0.75)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output1), m_pred3_gaussian2, 'r-', linewidth=4, alpha=0.5)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output1), m_pred3_gp_upper_gaussian2, 'r-', linewidth=2, alpha=1)
    plt.plot(np.sort(joint_Xtest_for_batch_3_output1), m_pred3_gp_lower_gaussian2, 'r-', linewidth=2, alpha=1)

    for q in range(model_list[2].Z.shape[1]):
        for m in range(model_list[2].Z[:,q].shape[0]):
            plt.axvline(model_list[2].Z[m, q], color='black', alpha=0.5)

    plt.title(r'Online Multi-Output Gaussian Regression (t=3)')
    plt.ylabel(r'Real Outputs')
    plt.xlabel(r'Real Inputs')
    plt.xlim(0, 1)
    #tikz_save('online_mogp_regression_batch_3.tex')
    plt.show()


# SINGLE OUTPUT (SO)
def plot_streaming_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    lik_noise = 1.5
    max_X = 2.0
    max_Y = 16.0
    T = len(model_list)
    for t in range(T):

        # For every batch in the stream:
        m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t],0), output_function_ind=0)
        m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new) + lik_noise
        m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new) - lik_noise

        fig_batch = plt.figure(figsize=(12, 4))

        if t>0:
            for t_past in range(t):
                m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past],0), output_function_ind=0)
                m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past) + 1.0
                m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past) - 1.0

                plt.plot(sXtrain[t_past], sYtrain[t_past], 'x', color='blue', markersize=10, alpha=0.25)
                plt.plot(sXtest[t_past], sYtest[t_past], 'o', color='blue', markersize=2, alpha=0.75)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_past, 'b', linewidth=4, alpha=0.5)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_gp_upper_past, 'b', linewidth=1, alpha=1)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_gp_lower_past, 'b', linewidth=1, alpha=1)


        plt.plot(sXtrain[t], sYtrain[t], 'x', color='red', markersize=10, alpha=0.2)
        plt.plot(sXtest[t], sYtest[t], 'o', color='red', markersize=2, alpha=0.75)
        plt.plot(np.sort(sXtest[t],0), m_pred_new, 'r', linewidth=4, alpha=0.5)
        plt.plot(np.sort(sXtest[t],0), m_pred_gp_upper_new, 'r', linewidth=1, alpha=1)
        plt.plot(np.sort(sXtest[t],0), m_pred_gp_lower_new, 'r', linewidth=1, alpha=1)

#        plt.plot(Z_points[t][:,0], q_mean_list[t], 'kx', markersize=15.0, mew=1.5)

        plt.xlim(0, max_X)
        plt.ylim(-max_Y, max_Y)
        plt.title(r'Continual Gaussian Process Regression (t=' + str(t+1) +')')
        plt.ylabel(r'Real Outputs')
        plt.xlabel(r'Real Inputs')

        if save:
            tikz_save('so_gpr_streaming_t'+str(t+1)+'.tex')

        plt.show()

# MULTI-OUTPUT (MO)
def plot_multioutput_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    max_X = 2.0
    max_Y = 13.0
    T = len(model_list)
    D = len(sXtrain[0])

    color_list = ['salmon', 'slateblue']
    new_color_list = ['red', 'blue']

    # For every batch in the stream:
    for t in range(T):

        fig_batch = plt.figure(figsize=(12, 4))

        # For every output in the model:
        for d in range(D):

            # For every batch in the stream:
            m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=d)
            m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new) + model_list[0].likelihood.likelihoods_list[d].sigma
            m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new) - model_list[0].likelihood.likelihoods_list[d].sigma

            if t>0:
                for t_past in range(t):
                    m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past][d], 0), output_function_ind=d)
                    m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past)  + model_list[0].likelihood.likelihoods_list[d].sigma
                    m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past)  - model_list[0].likelihood.likelihoods_list[d].sigma

                    plt.plot(sXtrain[t_past][d], sYtrain[t_past][d], 'x', color=color_list[d], markersize=10, alpha=0.25)
                    plt.plot(sXtest[t_past][d], sYtest[t_past][d], 'o', color=color_list[d], markersize=2, alpha=0.75)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_past, color=color_list[d], linewidth=4, alpha=0.5)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_upper_past, color=color_list[d], linewidth=1, alpha=1)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_lower_past, color=color_list[d], linewidth=1, alpha=1)

            plt.plot(sXtrain[t][d], sYtrain[t][d], 'x', color=new_color_list[d], markersize=10, alpha=0.2)
            plt.plot(sXtest[t][d], sYtest[t][d], 'o', color=new_color_list[d], markersize=2, alpha=0.75)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_new, color=new_color_list[d], linewidth=4, alpha=0.5)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_upper_new, color=new_color_list[d], linewidth=1, alpha=1)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_lower_new, color=new_color_list[d], linewidth=1, alpha=1)

        plt.xlim(0, max_X)
        plt.ylim(-max_Y, max_Y)
        plt.title(r'Continual Multi-output Gaussian Process Regression (t=' + str(t+1) +')')
        plt.ylabel(r'Real Outputs')
        plt.xlabel(r'Real Inputs')

        if save:
            tikz_save('mo_gpr_streaming_t'+str(t+1)+'.tex')

        plt.show()


# ASYNCHRONOUS MULTI-OUTPUT (MO)
def plot_asyn_multioutput_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    max_X = 1.0
    max_Y = 13.0
    T = len(model_list)
    D = len(sXtrain[0])

    color_list = ['salmon', 'slateblue']
    new_color_list = ['red', 'blue']

    # For every batch in the stream:
    for t in range(T):

        fig_batch = plt.figure(figsize=(12, 4))

        # For every output in the model:
        for d in range(D):

            # For every batch in the stream:
            m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=d)
            m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new) + model_list[0].likelihood.likelihoods_list[d].sigma
            m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new) - model_list[0].likelihood.likelihoods_list[d].sigma

            if t>0:
                for t_past in range(t):
                    m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past][d], 0), output_function_ind=d)
                    m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past)  + model_list[0].likelihood.likelihoods_list[d].sigma
                    m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past)  - model_list[0].likelihood.likelihoods_list[d].sigma

                    plt.plot(sXtrain[t_past][d], sYtrain[t_past][d], 'x', color=color_list[d], markersize=10, alpha=0.25)
                    plt.plot(sXtest[t_past][d], sYtest[t_past][d], 'o', color=color_list[d], markersize=2, alpha=0.75)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_past, color=color_list[d], linewidth=4, alpha=0.5)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_upper_past, color=color_list[d], linewidth=1, alpha=1)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_lower_past, color=color_list[d], linewidth=1, alpha=1)

            plt.plot(sXtrain[t][d], sYtrain[t][d], 'x', color=new_color_list[d], markersize=10, alpha=0.2)
            plt.plot(sXtest[t][d], sYtest[t][d], 'o', color=new_color_list[d], markersize=2, alpha=0.75)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_new, color=new_color_list[d], linewidth=4, alpha=0.5)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_upper_new, color=new_color_list[d], linewidth=1, alpha=1)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_lower_new, color=new_color_list[d], linewidth=1, alpha=1)

        plt.xlim(0, max_X)
        plt.ylim(-max_Y, max_Y)
        plt.title(r'Continual Multi-output Gaussian Process Regression (t=' + str(t+1) +')')
        plt.ylabel(r'Real Outputs')
        plt.xlabel(r'Real Inputs')

        if save:
            tikz_save('mo_gpr_asynchronous_t'+str(t+1)+'.tex')

        plt.show()

def plot_mocap_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    max_X = 1.0
    max_Y = 35.0
    T = len(model_list)
    D = len(sXtrain[0])

    color_list = ['lightpink', 'moccasin', 'thistle']
    new_color_list = ['crimson', 'orange', 'darkviolet']

    # For every batch in the stream:
    for t in range(T):

        fig_batch = plt.figure(figsize=(12, 4))

        # For every output in the model:
        for d in range(D):

            # For every batch in the stream:
            m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=d)
            m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new) + model_list[0].likelihood.likelihoods_list[d].sigma
            m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new) - model_list[0].likelihood.likelihoods_list[d].sigma

            if t>0:
                for t_past in range(t):
                    m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past][d], 0), output_function_ind=d)
                    m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past) + model_list[0].likelihood.likelihoods_list[d].sigma
                    m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past) - model_list[0].likelihood.likelihoods_list[d].sigma

                    plt.plot(sXtrain[t_past][d], sYtrain[t_past][d], 'x', color=color_list[d], markersize=10, alpha=1.0)
                    #plt.plot(sXtest[t_past][d], sYtest[t_past][d], 'o', color=color_list[d], markersize=2, alpha=0.75)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_past, color='k', linewidth=2, alpha=1.0)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_upper_past, color='k', linewidth=1, alpha=1.0)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_lower_past, color='k', linewidth=1, alpha=1.0)

            plt.plot(sXtrain[t][d], sYtrain[t][d], 'x', color=new_color_list[d], markersize=10, alpha=1.0)
            #plt.plot(sXtest[t][d], sYtest[t][d], 'o', color=new_color_list[d], markersize=2, alpha=0.75)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_new, color='k', linewidth=2, alpha=1.0)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_upper_new, color='k', linewidth=1, alpha=1.0)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_lower_new, color='k', linewidth=1, alpha=1.0)

        plt.xlim(0, max_X)
        plt.ylim(-max_Y, max_Y)
        plt.title(r'MOCAP (t=' + str(t+1) +')')
        plt.ylabel(r'Sensor Motion / Y axis output')
        plt.xlabel(r'Time')
        plt.legend(['Left Wrist Sensor', 'Right Femur Sensor', 'Mean Predictive Posterior'])

        if save:
            tikz_save('mocap_t'+str(t+1)+'.tex')

        plt.show()


# BANANA EXPERIMENT
def plot_banana_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, max_min, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    T = len(model_list)
    Ntest = 30
    min_X = max_min[0]
    min_Y = max_min[1]
    max_X = max_min[2]
    max_Y = max_min[3]

    stack_X_train = sXtrain[0]
    stack_Y_train = sYtrain[0]

    for t in range(T):

        max_ty = stack_X_train[:,1].max()
        min_ty = stack_X_train[:,1].min()
        max_tx = stack_X_train[:,0].max()
        min_tx = stack_X_train[:,0].min()

        ty = np.linspace(min_ty, max_ty, Ntest)
        tx = np.linspace(min_tx, max_tx, Ntest)
        TX_grid, TY_grid = np.meshgrid(tx, ty)
        TX = TX_grid.reshape(Ntest ** 2, 1)
        TY = TY_grid.reshape(Ntest ** 2, 1)
        test_X = np.hstack((TX, TY))

        # For every batch in the stream:
        m_pred_new, _ = model_list[t].predictive_new(test_X, output_function_ind=0)
        m_pred_new = np.exp(m_pred_new)/(1 + np.exp(m_pred_new))

        fig_batch = plt.figure(figsize=[8,6])

        if t > 0:
            for t_past in range(t):
                plt.plot(sXtrain[t_past][sYtrain[t_past][:, 0] == 1, 0], sXtrain[t_past][sYtrain[t_past][:, 0] == 1, 1], 'x', color='darkviolet', alpha=0.25)
                plt.plot(sXtrain[t_past][sYtrain[t_past][:, 0] == 0, 0], sXtrain[t_past][sYtrain[t_past][:, 0] == 0, 1], 'x', color='darkorange', alpha=0.25)

        plt.plot(sXtrain[t][sYtrain[t][:, 0] == 1, 0], sXtrain[t][sYtrain[t][:, 0] == 1, 1], 'x', color='darkviolet')
        plt.plot(sXtrain[t][sYtrain[t][:, 0] == 0, 0], sXtrain[t][sYtrain[t][:, 0] == 0, 1], 'x', color='darkorange')

        plt.contour(TX_grid, TY_grid, np.reshape(m_pred_new, (Ntest, Ntest)), linewidths=3, colors='k', levels=0.5 * np.eye(1))

        plt.xlim(min_X, max_X)
        plt.ylim(min_Y, max_Y)
        plt.title(r'Continual Gaussian Process Classification (t=' + str(t + 1) + ')')
        plt.ylabel(r'Real Input')
        plt.xlabel(r'Real Input')
        plt.legend(['y=1', 'y=-1'])

        if save:
            tikz_save('banana_t' + str(t + 1) + '.tex')

        plt.show()

        if t < T-1:
            stack_X_train = np.vstack((stack_X_train, sXtrain[t+1]))
            stack_Y_train = np.vstack((stack_Y_train, sYtrain[t+1]))


    return

# CURRENCY EXPERIMENT
def plot_currency_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, mean_usd, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    lik_noise = model_list[0].likelihood.likelihoods_list[0].sigma
    max_X = 2.0
    max_Y = 0.12
    #max_Y = 16.0
    T = len(model_list)

    for t in range(T):

        # For every batch in the stream:
        m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t],0), output_function_ind=0)
        m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new) + lik_noise
        m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new) - lik_noise

        fig_batch = plt.figure(figsize=(12, 4))

        if t>0:
            for t_past in range(t):
                m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past],0), output_function_ind=0)
                m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past) + lik_noise
                m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past) - lik_noise

                plt.plot(sXtrain[t_past][sXtrain[t_past][:,0].argsort()], sYtrain[t_past][sXtrain[t_past][:,0].argsort()]+mean_usd, '-', color='steelblue', linewidth=2, markersize=10, alpha=1.0)
                #plt.plot(sXtest[t_past][sXtest[t_past][:,0].argsort()], sYtest[t_past][sXtest[t_past][:,0].argsort()], '-', color='blue', markersize=2, alpha=0.75)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_past+mean_usd, '-k', linewidth=2, alpha=1.0)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_gp_upper_past, '-k', linewidth=1, alpha=1.0)
                plt.plot(np.sort(sXtest[t_past],0), m_pred_gp_lower_past, '-k', linewidth=1, alpha=1.0)


        plt.plot(sXtrain[t][sXtrain[t][:,0].argsort()], sYtrain[t][sXtrain[t][:,0].argsort()]+mean_usd, '-', color='mediumturquoise', linewidth=2, markersize=10, alpha=1.0)
        #plt.plot(sXtest[t][sXtest[t][:,0].argsort()], sYtest[t][sXtest[t][:,0].argsort()], '-', color='red', markersize=2, alpha=0.75)
        plt.plot(np.sort(sXtest[t],0), m_pred_new+mean_usd, '-k', linewidth=2, alpha=1.0)
        plt.plot(np.sort(sXtest[t],0), m_pred_gp_upper_new, '-k', linewidth=1, alpha=1.0)
        plt.plot(np.sort(sXtest[t],0), m_pred_gp_lower_new, '-k', linewidth=1, alpha=1.0)

        plt.xlim(0, max_X)
        plt.ylim(-max_Y, max_Y)
        plt.title(r'Dollar Exchange Rate (t=' + str(t+1) +')')
        plt.ylabel(r'USD/EUR')
        plt.xlabel(r'Real Inputs')

        if save:
            tikz_save('currency_t'+str(t+1)+'.tex')

        plt.show()

# BANANA MSE AND NLPD
def banana_metrics(model_list, sXtest, sYtest):
    T = len(model_list)
    for t in range(T):
        m_pred_test, _ = model_list[t].predictive_new(sXtest[t], output_function_ind=0)
        m_pred_test = np.exp(m_pred_test) / (1 + np.exp(m_pred_test))
        m_pred_test[m_pred_test[:,0]<0.5, 0] = 0.0
        m_pred_test[m_pred_test[:,0]>=0.5, 0] = 1.0

        errors = sYtest[t] - m_pred_test
        errors = np.abs(errors)
        mse = np.sum(errors,axis=0)/errors.shape[0]

        print(mse)

    return

def plot_eb2_latex(model_list, sXtrain, sXtest, sYtrain, sYtest, Z_points, q_mean_list, save=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    max_X = 1.0
    max_Y = 35.0
    T = len(model_list)
    D = len(sXtrain[0])

    color_list = ['lightpink', 'moccasin', 'thistle']
    new_color_list = ['crimson', 'orange', 'darkviolet']

    # For every batch in the stream:
    for t in range(T):

        # For every output in the model:
        for d in range(D):

            fig_batch = plt.figure(figsize=(12, 4))

            if d == 0:
                # For every batch in the stream:
                m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=0)
                m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(v_pred_new)
                m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(v_pred_new)

                m_pred_new = np.exp(m_pred_new) / (1 + np.exp(m_pred_new))
                m_pred_gp_upper_new = np.exp(m_pred_gp_upper_new) / (1 + np.exp(m_pred_gp_upper_new))
                m_pred_gp_lower_new = np.exp(m_pred_gp_lower_new) / (1 + np.exp(m_pred_gp_lower_new))

            elif d == 1:
                m_pred_new, v_pred_new = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=1)
                m_pred_new_v, v_pred_new_v = model_list[t].predictive_new(np.sort(sXtest[t][d], 0), output_function_ind=2)
                m_pred_gp_upper_new = m_pred_new + 2 * np.sqrt(np.exp(m_pred_new_v))
                m_pred_gp_lower_new = m_pred_new - 2 * np.sqrt(np.exp(m_pred_new_v))

            if t>0:
                for t_past in range(t):
                    if d==0:
                        m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past][d],0),output_function_ind=0)
                        m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(v_pred_past)
                        m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(v_pred_past)

                        m_pred_past = np.exp(m_pred_past) / (1 + np.exp(m_pred_past))
                        m_pred_gp_upper_past = np.exp(m_pred_gp_upper_past) / (1 + np.exp(m_pred_gp_upper_past))
                        m_pred_gp_lower_past = np.exp(m_pred_gp_lower_past) / (1 + np.exp(m_pred_gp_lower_past))
                    elif d==1:
                        m_pred_past, v_pred_past = model_list[t].predictive_new(np.sort(sXtest[t_past][d],0),output_function_ind=1)
                        m_pred_past_v, v_pred_past_v = model_list[t].predictive_new(np.sort(sXtest[t_past][d],0),output_function_ind=2)
                        m_pred_gp_upper_past = m_pred_past + 2 * np.sqrt(np.exp(m_pred_past_v))
                        m_pred_gp_lower_past = m_pred_past - 2 * np.sqrt(np.exp(m_pred_past_v))

                    plt.plot(sXtrain[t_past][d], sYtrain[t_past][d], 'x', color=color_list[d], markersize=10, alpha=1.0)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_past, color='k', linewidth=2, alpha=1.0)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_upper_past, color='k', linewidth=1, alpha=1.0)
                    plt.plot(np.sort(sXtest[t_past][d], 0), m_pred_gp_lower_past, color='k', linewidth=1, alpha=1.0)

            plt.plot(sXtrain[t][d], sYtrain[t][d], 'x', color=new_color_list[d], markersize=10, alpha=1.0)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_new, color='k', linewidth=2, alpha=1.0)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_upper_new, color='k', linewidth=1, alpha=1.0)
            plt.plot(np.sort(sXtest[t][d], 0), m_pred_gp_lower_new, color='k', linewidth=1, alpha=1.0)

            plt.xlim(0, 1.0)

            plt.title(r'Human Behavior (t=' + str(t+1) +')')
            plt.ylabel(r'Heterogeneous outputs')
            plt.xlabel(r'Time')

            if save:
                tikz_save('behavior_t'+str(t+1)+'_d'+str(d+1)+'.tex')

        plt.show()