# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np
import scipy.sparse as sp
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam

import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
import cost
from get_inputs import get_inputs

import pdb

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0, 1, 2, 4, 5]
NBASISTENSORS = 10
NSCALARINVARIANTS = 5

VECTORDIM = 3


class Model(PhysicsModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, inputs_dafi, inputs_model):
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']

        nweights = inputs_model.get('nweights', None)
        self.ncpu = inputs_model.get('ncpu', 20)
        self.rel_stddev = inputs_model.get('rel_stddev', 0.5)
        self.abs_stddev = inputs_model.get('abs_stddev', 0.5)
        self.obs_rel_std = inputs_model.get('obs_rel_std', 0.001)
        self.obs_abs_std = inputs_model.get('obs_abs_std', 0.0001)

        obs_file = inputs_model['obs_file']

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.da_iteration = -1

        # # control dictionary
        self.timeprecision = 6

        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)

        self.nstate = len(self.w_init)

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        # observations
        # read experimental observations

        expArray = np.loadtxt(obs_file + '/Array-exp.dat')

        self.obs = expArray
        self.obs_error = np.diag(self.obs_rel_std * abs(self.obs) +
                                 self.obs_abs_std)
        self.nstate_obs = len(self.obs)

        # create sample directories
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            shutil.copytree(self.foam_case, sample_dir)
        self.sample_dirs = sample_dirs

    def __str__(self):
        return 'Dynamic model for nutFoam eddy viscosity solver.'

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Creates the OpenFOAM case directories for each sample, creates
        samples of eddy viscosity (nut) based on samples of the KL modes
        coefficients (state) and writes nut field files. Returns the
        coefficients of KL modes for each sample.
        """

        # update X
        w = np.zeros([self.nstate, self.nsamples])
        for i in range(self.nstate):
            w[i, :] = self.w_init[i] + np.random.normal(
                0, abs(self.w_init[i] * self.rel_stddev + self.abs_stddev),
                self.nsamples)
        return w

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.
        """
        self.da_iteration += 1

        # set weights
        w = state_vec.copy()

        gsamps = []
        ts = time.time()
        for isamp in range(self.nsamples):
            modelPath = os.path.join(self._sample_dir(isamp),
                                     'nn_weights_flatten.dat')
            np.savetxt(modelPath, w[:, isamp])
            modelPath1 = os.path.join(self._sample_dir(isamp),
                                     'nn_weights_flatten_' + str(self.da_iteration+2) + '.dat')
            np.savetxt(modelPath1, w[:, isamp])
        print(self.da_iteration, 'nn_weight had saved')

        parallel = multiprocessing.Pool(self.ncpu)
        inputs = [(self._sample_dir(i), self.da_iteration, self.timeprecision)
                  for i in range(self.nsamples)]
        _ = parallel.starmap(_run_foam, inputs)
        parallel.close()

        # get HX
        time_start = 1
        time_step = 1

        time_dir = f'{((self.da_iteration+1)*time_step+time_start):g}'
        state_in_obs = np.empty([self.nstate_obs, self.nsamples])
        for isample in range(self.nsamples):

            file = os.path.join(self._sample_dir(isample), 'postProcessing',
                                'sampleDict', time_dir)

            if os.path.exists(file + '/Array-obs.dat'):
                state_in_obs[:, isample] = np.loadtxt(file + '/Array-obs.dat')
            else:
                time_dir_last = f'{((self.da_iteration+1)*time_step+time_start-1):g}'
                file_last = os.path.join(self._sample_dir(isample), 'postProcessing',
                                'sampleDict', time_dir_last)
                state_in_obs[:, isample] = np.loadtxt(file_last + '/Array-obs.dat')
                state_vec_path = os.path.join(self._sample_dir(isamp),'nn_weights_flatten_' + str(self.da_iteration+1) + '.dat')
                state_vec[:, isample] = np.loadtxt(state_vec_path)

        print('state_in_obs using', time.time() - ts)
        return state_in_obs, state_vec

    def get_obs(self, time):
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')


def _run_foam(foam_dir, iteration, timeprecision):
    # run foam
    os.chdir(foam_dir)

    bash_command = "./run.sh"
    subprocess.call(bash_command, shell=True)

    os.chdir('../../')
