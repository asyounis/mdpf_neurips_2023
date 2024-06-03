# Standard Imports
import os
import numpy as np
import math
import random
import shutil
import PIL
# from PIL import Image
from tqdm import tqdm
import cv2
from matplotlib import colors


# Pytorch Imports
import torch
import torch.distributions as D

# Project imports
from utils import *
from datasets.base_dataset import *


class KalmanFilter:
    def __init__(self, measurement_std, process_std):

        self.measurement_std = measurement_std
        self.process_std = process_std

    def predict_step(self, filter_state, action):

        # Unpack
        x, sqrt_p = filter_state

        # Compute P
        p = sqrt_p**2

        # Evolve the state
        x = x + action

        # Evolve the variance
        p = p +  (self.process_std**2)

        # return a repacked state
        return (x, torch.sqrt(p))

    def measuremnt_step(self, filter_state, measurement):

        # Unpack
        x, sqrt_p = filter_state

        # Compute P
        p = sqrt_p**2

        # Compute Kalman Gain
        k = p / (p + self.measurement_std**2)

        # Update state and variance
        x = x + (k * (measurement - x))
        p = (1-k) * p

        # return a repacked state
        return (x, torch.sqrt(p))

    def generate_full_estimate(self, starting_filter_state, actions, measurements):

        # Kalman Filter State: (mean, std)
        filter_state = starting_filter_state

        # Keep track of the states
        all_filter_states = []
        all_filter_states.append(filter_state)

        # Run the filter
        for i in range(measurements.shape[1] - 1):
            
            # Extract the data
            measurement = measurements[:, i+1]
            action = actions[:, i]

            # Do an update step
            filter_state = self.predict_step(filter_state, action)

            # Do an update step
            filter_state = self.measuremnt_step(filter_state, measurement)

            # Store the state for plotting later
            all_filter_states.append(filter_state)

        # Need to do some processing to get them into tensors
        all_filter_states = [torch.stack(fs, dim=-1)  for fs in all_filter_states]
        all_filter_states = torch.stack(all_filter_states, axis=1)

        return all_filter_states


class SimpleBrownianMotionTrackingDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):
        
        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # Extract the general parameters for this Dataset
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")

        # Extract the size of the dataset for this dataset
        dataset_sizes = get_parameter_safely("dataset_sizes", dataset_params, "dataset_params")
        self.dataset_size = get_parameter_safely(dataset_type, dataset_sizes, "dataset_sizes")

        # Get the dataset save location
        dataset_saves = get_parameter_safely("dataset_saves", dataset_params, "dataset_params")
        self.save_location = get_parameter_safely(dataset_type, dataset_saves, "dataset_saves")

        # If the directory does not exist then we want to create it!
        data_dir, _ = os.path.split(self.save_location)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Load some dataset parameters
        self.delta = get_parameter_safely("delta", dataset_params, "dataset_params")
        self.dt = get_parameter_safely("dt", dataset_params, "dataset_params")
        self.measurement_std = get_parameter_safely("measurement_std", dataset_params, "dataset_params")
        self.action_std = get_parameter_safely("action_std", dataset_params, "dataset_params")


        # try to load the data and if not then generate the data
        # if(self._load_dataset() == False):
        if(True):
        
            # Generate the dataset
            self._generate_dataset()

            # Save the dataset parameters if we can
            self._save_dataset()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        # Generate the return dictionary
        return_dict = {}
        return_dict["dataset_index"] = idx
        return_dict["states"] = self.all_true_states[idx].float()
        return_dict["observations"] = self.all_measurements[idx].float()
        return_dict["actions"] = self.all_actions[idx].float()
        return_dict["kalman_filter_estimate"] = self.all_kalman_filter_estimates[idx].float()

        return return_dict

    def get_subsequence_length(self):
        return self.subsequence_length
        
    # def get_x_range(self):
    #     return (-10, 10)

    # def get_y_range(self):
    #     return (-10, 10)

    def _save_dataset(self):


        save_dict = dict()
        save_dict["dataset_params"] = self.dataset_params
        save_dict["dataset_type"] = self.dataset_type

        save_dict["all_true_states"] = self.all_true_states
        save_dict["all_actions"] = self.all_actions
        save_dict["all_measurements"] = self.all_measurements
        save_dict["all_kalman_filter_estimates"] = self.all_kalman_filter_estimates

        # Save that dict
        torch.save(save_dict, self.save_location)


    def _load_dataset(self):

        # Check if the file exists, if not then we cant load 
        if(not os.path.exists(self.save_location)):
            return False

        # Load and unpack data
        loaded_data = torch.load(self.save_location)

        if(loaded_data["dataset_params"] != self.dataset_params):
            return False

        if(loaded_data["dataset_type"] != self.dataset_type):
            return False


        # Unpack
        self.all_true_states = loaded_data["all_true_states"] 
        self.all_actions = loaded_data["all_actions"] 
        self.all_measurements = loaded_data["all_measurements"] 
        self.all_kalman_filter_estimates = loaded_data["all_kalman_filter_estimates"] 

        return True


    def _generate_dataset(self):

        # All the data we want to generate 
        all_true_states = []
        all_actions = []
        all_measurements = []

        # Generate all the sequences
        for seq_num in tqdm(range(self.dataset_size)):  

            # Generate a sequence and save it
            true_states, actions, measurements = self._generate_sequence()
            all_true_states.append(true_states)
            all_actions.append(actions)
            all_measurements.append(measurements)

        # Convert to numpy
        self.all_true_states = np.asarray(all_true_states)
        self.all_actions = np.asarray(all_actions)
        self.all_measurements = np.asarray(all_measurements)

        # Convert to pytorch
        self.all_true_states = torch.from_numpy(self.all_true_states)
        self.all_actions = torch.from_numpy(self.all_actions)
        self.all_measurements = torch.from_numpy(self.all_measurements)

        # Need to make 1-D instead of "0-D"
        self.all_true_states = self.all_true_states.unsqueeze(-1)
        self.all_actions = self.all_actions.unsqueeze(-1)
        self.all_measurements = self.all_measurements.unsqueeze(-1)


        # Generate the optimal estimate using an optimal estimator (Kalman filter)
        self._generate_kalman_filter_solution()

    def _generate_sequence(self):

        # generate the true states
        x0 = np.random.uniform(low=-50, high=50, size=(1, ))
        true_states = self._generate_brownian(x0=x0, n=self.subsequence_length, dt=self.dt, delta=self.delta)
        true_states = true_states[0]

        # Compute the actions (aka velocities)
        actions = np.zeros(true_states.shape)
        actions[0:-1] = true_states[1:] - true_states[:-1]
        actions += (np.random.randn(actions.shape[0]) * self.action_std)

        # Compute the measurements
        measurements = true_states + (np.random.randn(true_states.shape[0])*self.measurement_std)

        return true_states, actions, measurements


    def _generate_brownian(self, x0, n, dt, delta):
        """
        Generate an instance of Brownian motion (i.e. the Wiener process):

            X(t) = X(0) + N(0, delta**2 * t; 0, t)

        where N(a,b; t0, t1) is a normally distributed random variable with mean a and
        variance b.  The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
        are independent.
        
        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


        If `x0` is an array (or array-like), each value in `x0` is treated as
        an initial condition, and the value returned is a numpy array with one
        more dimension than `x0`.

        Arguments
        ---------
        x0 : float or numpy array (or something that can be converted to a numpy array
             using numpy.asarray(x0)).
            The initial condition(s) (i.e. position(s)) of the Brownian motion.
        n : int
            The number of steps to take.
        dt : float
            The time step.
        delta : float
            delta determines the "speed" of the Brownian motion.  The random variable
            of the position at time t, X(t), has a normal distribution whose mean is
            the position at time t=0 and whose variance is delta**2*t.

        Returns
        -------
        A numpy array of floats with shape `x0.shape + (n,)`.
        
        Note that the initial value `x0` is not included in the returned array.
        """
        x0 = np.asarray(x0)

        # For each element of x0, generate a sample of n numbers from a normal distribution.
        shape = list(x0.shape)
        shape.append(n-1)
        r = np.random.randn(*shape) * delta * np.sqrt(dt)

        shape = list(x0.shape)
        shape.append(n)        
        out = np.zeros(shape)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples. 
        out[..., 1:] = np.cumsum(r, axis=-1)

        # Add the initial condition.
        out += np.expand_dims(x0, axis=-1)

        return out




    def _generate_kalman_filter_solution(self):

        # Make a Kalman Filter Object
        kf = KalmanFilter(self.measurement_std, self.action_std)

        # get rid of the last dim 
        true_states = self.all_true_states[..., 0]
        actions = self.all_actions[..., 0]
        measurements = self.all_measurements[..., 0]

        # Make the starting state for the filter (the mean and variance of the filter)
        starting_filter_state = (true_states[:, 0], torch.ones(true_states[:, 0].shape) * 10.0)

        # Run the filter
        self.all_kalman_filter_estimates = kf.generate_full_estimate(starting_filter_state, actions, measurements)


