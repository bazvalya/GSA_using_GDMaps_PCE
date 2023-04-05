# GDMaps, PceModel, ErrorEstimation are taken from and built upon https://github.com/katiana22/GDM-PCE

import numpy as np
import time
import os, subprocess
import random

import matplotlib.pyplot as plt
import matplotlib as mpl

from UQpy.surrogates import *
from UQpy.sensitivity import *

from sklearn.model_selection import train_test_split

from DimensionReduction import Grassmann
from DimensionReduction import DiffusionMaps


#######################################################################################################################
#######################################################################################################################
#                                                   GDMaps PCE GSA                                                    #
#######################################################################################################################
#######################################################################################################################

class GDMaps:
    """
    Performs GDMaps for a given dataset.
    n_evecs must be greater than n_keep
    """

    def __init__(self, data, n_evecs, n_keep, p, parsim=True, verbose=False):
        self.data = data
        self.n_evecs = n_evecs
        self.n_keep = n_keep
        self.p = p
        self.parsim = parsim
        self.verbose = verbose

    def get(self):
        Gr = Grassmann(distance_method=Grassmann.grassmann_distance, 
                       kernel_method=Grassmann.projection_kernel,
                       karcher_method=Grassmann.gradient_descent)
        Gr.manifold(p=self.p, samples=self.data)

        dfm = DiffusionMaps(alpha=0.5, 
                            n_evecs=self.n_evecs + 1, 
                            kernel_object=Gr, 
                            kernel_grassmann='prod')
        g, evals, evecs = dfm.mapping()
        
#         print('Grassmann projection rank is: ', Gr.p)
        
        if self.parsim:
            print('Running with parsimonious representation')

            # Residuals used to identify the most parsimonious low-dimensional representation.
            index, residuals = dfm.parsimonious(num_eigenvectors=self.n_evecs, visualization=False)
            
            coord = index[1:self.n_keep + 1]
            g_k = g[:, coord]
            
            return g_k, coord, Gr, residuals, index, evals, evecs
            
        else:
            print(f'Keeping first {self.n_keep} nontrivial eigenvectors')
            
            coord = np.arange(1, self.n_keep+1) # keeping first n_keep nontrivial eigenvectors
            g_k = g[:, coord]
            
            return g_k, coord, Gr, evals, evecs

            
    
class PceModel:
    """
    Constructs a PCE surrogate on the Grassmannian diffusion manifold.
    """

    def __init__(self, x, g, dist_obj, max_degree, regression='OLS', verbose=False):
        self.x = x
        self.g = g
        self.dist_obj = dist_obj
        self.max_degree = max_degree
        self.regression = regression
        self.verbose = verbose

    def get(self):

        # Polynomial basis
        polynomial_basis = TotalDegreeBasis(distributions=self.dist_obj, 
                                            max_degree=self.max_degree)
        
        # Regression
        if self.regression == 'OLS':
            reg = LeastSquareRegression()
        
        elif self.regression == 'Lasso':
            # function parameters need to be tuned
            reg = LassoRegression(learning_rate=0.001, iterations=1000, penalty=0.05)
            
        elif self.regression == 'Ridge':
            # function parameters need to be tuned
            reg = RidgeRegression(learning_rate=0.001, iterations=10000, penalty=0)
            
        else:
            raise ValueError('The only allowable input strings are `OLS`, `Lasso`, and `Ridge`.')
        
        pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=reg)

        x_train, x_test, \
        g_train, g_test = train_test_split(self.x, self.g, train_size=2 / 3, random_state=1)
        
        # Fit model
        pce.fit(x_train, g_train)
        
        print('Size of the full set of PCE basis:', pce.polynomial_basis.polynomials_number)
        print('Shape of the training set (x)):', x_train.shape)
        print('Shape of the training set (y)):', g_train.shape)
    
        error_val = ErrorEstimation(surr_object=pce).validation(x_test, g_test)

        if self.verbose:
            # Plot accuracy of PCE
            if os.path.exists('pce_accuracy'):
                command = ['rm', '-r', 'pce_accuracy']
                subprocess.run(command)

            command = ['mkdir', 'pce_accuracy']
            subprocess.run(command)

            print(g_test[0, :])
            print(pce.predict(x_test)[0, :])

            for i in range(5):
                r = random.randint(0, x_test.shape[0])
                plt.figure()
                plt.plot(g_test[r, :], 'b-o', label='true')
                plt.plot(pce.predict(x_test)[r, :], 'r-*', label='pce')
                plt.legend()
                plt.savefig('pce_accuracy/pce_{}.png'.format(i), bbox_inches='tight')
                plt.show()

        return pce, error_val

       

class ErrorEstimation:
    """
    Class for estimating the error of a PCE surrogate, based on a validation
    dataset. Used in PceModel

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    def __init__(self, surr_object):
        self.surr_object = surr_object

    def validation(self, x, y):
        """
        Returns the validation error.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the samples of the validation dataset.

        * **y** (`ndarray`):
            `ndarray` containing model evaluations for the validation dataset.

        **Outputs:**

        * **eps_val** (`float`)
            Validation error.

        """
        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        y_val = self.surr_object.predict(x)

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = (n_samples - 1) / n_samples * (
                (np.sum((y - y_val) ** 2, axis=0)) / (np.sum((y - mu_yval) ** 2, axis=0)))

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)
    


def run_PCE_GSA_on_mainfold(p, data, x, num_runs, num_vars, n_keep, dist_obj, pce_max_degree=15, parsim=False):
    """Runs Global Sensitivity Analysis using PCE surrogate on manifold for each of runs"""   
    n2 = data[0].shape[1]
    
    n_d_coord = n_keep
    
    # Arrays with Sobol indices
    pce_total_Si = np.zeros((num_runs, num_vars, n_d_coord))
    pce_first_Si = np.zeros((num_runs, num_vars, n_d_coord))
    
    # Arrays with generalised Sobol indices
    pce_gto_Si = np.zeros((num_runs, num_vars))
    pce_gfo_Si = np.zeros((num_runs, num_vars))
    
    evals_diff_runs  = []
    evecs_diff_runs  = []
    coord_diff_runs = []
    g_diff_runs = []
    
    pce_error_diff_runs = []
    

    for i in range(num_runs):
        print('Run: ', i)
        data_all = data[i].reshape(-1, 
                                   int(np.sqrt(n2)), 
                                   int(np.sqrt(n2)))

        # perform GDMAps
        start_time = time.time()
            
        g, coord, Grass, evals, evecs = GDMaps(data=data_all, 
                                               n_evecs=20,
                                               n_keep=n_keep,
                                               parsim=parsim,
                                               p=p).get()

        evals_diff_runs.append(evals)
        evecs_diff_runs.append(evecs)
        coord_diff_runs.append(coord)
        g_diff_runs.append(g)
        
        print("--- GDMaps - %s seconds ---" % (time.time() - start_time))
        
        # perform PCE on the manifold
        start_time = time.time()
        pce, error = PceModel(x=x, 
                              g=g, 
                              dist_obj=dist_obj, 
                              max_degree=pce_max_degree,
                              verbose=False).get()
        
        pce_error_diff_runs.append(error)
        
        print('Error of PCE:', error)
        print("--- PCE surrogate - %s seconds ---" % (time.time() - start_time))

        pce_to  = PceSensitivity(pce).calculate_total_order_indices()
        pce_fo  = PceSensitivity(pce).calculate_first_order_indices()
        pce_gto = PceSensitivity(pce).calculate_generalized_total_order_indices()
        pce_gfo = PceSensitivity(pce).calculate_generalized_first_order_indices()

        for param in range(num_vars):
            pce_total_Si[i, param, :] = pce_to[param]
            pce_first_Si[i, param, :] = pce_fo[param]

        pce_gto_Si[i] = pce_gto
        pce_gfo_Si[i] = pce_gfo
        
    return (pce_total_Si, pce_first_Si, pce_gto_Si, pce_gfo_Si,
            evals_diff_runs, evecs_diff_runs, coord_diff_runs, g_diff_runs,
            pce_error_diff_runs)
