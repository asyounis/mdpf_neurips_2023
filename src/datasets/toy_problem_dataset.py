# Standard Imports
import os
import numpy as np
import math

# Useful Imports
import PIL
from PIL import Image
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.distributions as D

# Project imports
from utils import *
from datasets.base_dataset import *

class ToyProblemDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):

        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # Extract the params        
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")

        # Extract the size of the dataset for this dataset
        dataset_sizes = get_parameter_safely("dataset_sizes", dataset_params, "dataset_params")
        self.dataset_size = get_parameter_safely(dataset_type, dataset_sizes, "dataset_sizes")

        # See if we should use sparse ground truths
        if("sparse_ground_truth_keep_modulo" in dataset_params):
            self.sparse_ground_truth_keep_modulo = dataset_params["sparse_ground_truth_keep_modulo"]
        else:
            self.sparse_ground_truth_keep_modulo = None

        # check if we can load a dataset
        did_load_dataset = self.load_from_save(dataset_params, dataset_type)

        # Generate the dataset if we did not load one
        if(not did_load_dataset):
            self.generate_sequences(self.dataset_size, self.subsequence_length)

            # Save it, this will do nothing if we are unable to save 
            self.save_dataset()

    def load_from_save(self, dataset_params, dataset_type):

        # The dafault save location is none (aka dont save)
        self.save_location = None

        # Check if we have the save params, if not then we cant load from save
        if("dataset_saves" not in dataset_params):
            return False

        # Extract this dataset save location 
        dataset_saves = get_parameter_safely("dataset_saves", dataset_params, "dataset_params")

        # If this dataset type is not in the list of saved datasets then we cannot save or load this specific dataset
        if(dataset_type not in dataset_saves):
            return False

        # Extract the save location
        self.save_location = get_parameter_safely(dataset_type, dataset_saves, "dataset_saves")

        # If the directory does not exist then we want to create it!
        data_dir, _ = os.path.split(self.save_location)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Check if the file exists, if not then we cant load 
        if(not os.path.exists(self.save_location)):
            return False

        # Load and unpack data
        all_data = torch.load(self.save_location)
        states = all_data["states"]
        observations = all_data["observations"]
        start_time = all_data["start_time"]

        # Check the data to make sure that it is correct
        if(states.shape[0] != self.dataset_size):
            return False
        if(states.shape[1] != self.subsequence_length):
            return False

        # The data looks good so we should use it
        self.states = states
        self.observations = observations
        self.start_time = start_time

        # We have successfully loaded the dataset
        return True

    def save_dataset(self):

        # If we dont have a save location then we cant save
        if(self.save_location is None):
            return

        # Pack everything into a single dict that we can save
        save_dict = dict()
        save_dict["states"] = self.states
        save_dict["observations"] = self.observations
        save_dict["start_time"] = self.start_time

        # Save that dict
        torch.save(save_dict, self.save_location)



    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        # Compute the return dictionary
        return_dict = {}
        return_dict["states"] = self.states[idx]
        return_dict["observations"] = self.observations[idx]
        return_dict["start_time"] = self.start_time[idx]
        return_dict["dataset_index"] = idx


        if(self.sparse_ground_truth_keep_modulo is not None):
            ground_truth_mask = torch.full(size=(self.states[idx].shape[0],), fill_value=False)

            for i in range(ground_truth_mask.shape[0]):
                if((i % self.sparse_ground_truth_keep_modulo) == 0):
                    ground_truth_mask[i] = True

            return_dict["ground_truth_mask"] = ground_truth_mask


        return return_dict

    def get_range(self):
        return (-30, 30)


    def get_subsequence_length(self):
        return self.subsequence_length

    # def generate_sequences(self, dataset_size, subsequence_length):
    #     ''' Implement the dyamical model from Example 1 for the paper:
    #             "An Overview of Existing Methods and Recent Advances in Sequential Monte Carlo" by Olivier Cappe, Simon J. Godsill, and Eric Moulines
    #     ''' 

    #     # Constants from the paper
    #     SIGMA_U = math.sqrt(10)
    #     SIGMA_V = math.sqrt(1)

    #     # The states and the observations we will be generating
    #     self.states = torch.zeros(size=(dataset_size, subsequence_length))
    #     self.observations = torch.zeros(size=(dataset_size, subsequence_length))

    #     # Generate the first state
    #     initial_dist = D.Normal(0, SIGMA_U)
    #     self.states[:, 0] = initial_dist.sample(sample_shape=(dataset_size,))

    #     # Generate the rest of the states
    #     for i in range(1, subsequence_length):
            
    #         # Extract the previous x
    #         x_prev = self.states[:, i-1].clone()

    #         # Generate the mean for the next x that will be on the 
    #         x_mean = (x_prev / 2.0)
    #         x_mean += 25.0 * (x_prev / (1 + (x_prev**2)))
    #         x_mean += 8 * math.cos(1.2 * i)

    #         # Generate the sampling Distribution
    #         new_x_dist = D.Normal(x_mean, SIGMA_U)

    #         # Sample
    #         self.states[:, i] = new_x_dist.sample()


    #     # Generate the observations
    #     observation_mean = self.states.clone()
    #     observation_mean = (observation_mean**2) / 20.0

    #     # Generate the sampling Distribution
    #     observation_dist = D.Normal(observation_mean, SIGMA_V)

    #     # Sample
    #     self.observations[...] = observation_dist.sample()
    #     # self.observations[...] = observation_mean

    #     # Make the tensor have the correct dims
    #     self.states = self.states.unsqueeze(-1)
    #     self.observations = self.observations.unsqueeze(-1)

    #     self.start_time = torch.zeros(size=(dataset_size, 1))


    def generate_sequences(self, dataset_size, subsequence_length):
        ''' Implement the dyamical model from Example 1 for the paper:
                "An Overview of Existing Methods and Recent Advances in Sequential Monte Carlo" by Olivier Cappe, Simon J. Godsill, and Eric Moulines
        ''' 

        # Constants from the paper
        SIGMA_U = math.sqrt(10)
        SIGMA_V = math.sqrt(1)


        # NUMBER_OF_LONG_SEQUENCES = int(dataset_size / 2)
        # NUMBER_OF_LONG_SEQUENCES = int(dataset_size)
        NUMBER_OF_LONG_SEQUENCES = int(100)
        LONG_SEQUENCE_LENGTH = int(dataset_size * subsequence_length / NUMBER_OF_LONG_SEQUENCES)


        long_sequence_states = torch.zeros(size=(NUMBER_OF_LONG_SEQUENCES, LONG_SEQUENCE_LENGTH))
        long_sequence_obs = torch.zeros(size=(NUMBER_OF_LONG_SEQUENCES, LONG_SEQUENCE_LENGTH))

        # Generate the first state
        initial_dist = D.Normal(0, SIGMA_U)
        long_sequence_states[:, 0] = initial_dist.sample(sample_shape=(NUMBER_OF_LONG_SEQUENCES,))

        # Generate the rest of the states
        print("Generating Data")
        for i in tqdm(range(1, LONG_SEQUENCE_LENGTH)):
            
            # Extract the previous x
            x_prev = long_sequence_states[:, i-1].clone()

            # Generate the mean for the next x that will be on the 
            x_mean = (x_prev / 2.0)
            x_mean += 25.0 * (x_prev / (1 + (x_prev**2)))
            x_mean += 8 * math.cos(1.2 * i)

            # Generate the sampling Distribution
            new_x_dist = D.Normal(x_mean, SIGMA_U)

            # Sample
            long_sequence_states[:, i] = new_x_dist.sample()

        # Generate the observations
        observation_mean = long_sequence_states.clone()
        observation_mean = (observation_mean**2) / 20.0

        # Generate the sampling Distribution
        observation_dist = D.Normal(observation_mean, SIGMA_V)

        # Sample
        long_sequence_obs[...] = observation_dist.sample()


        long_sequence_states = long_sequence_states.reshape(-1)
        long_sequence_obs = long_sequence_obs.reshape(-1)

        self.states = torch.zeros(size=(dataset_size, subsequence_length))
        self.observations = torch.zeros(size=(dataset_size, subsequence_length))
        self.start_time = torch.zeros(size=(dataset_size, 1))
        for i in range(dataset_size):
            s = i * subsequence_length
            e = s + subsequence_length

            self.states[i,:] = long_sequence_states[s:e]
            self.observations[i,:] = long_sequence_obs[s:e]
            self.start_time[i] = float(s % LONG_SEQUENCE_LENGTH)

        # Make the tensor have the correct dims
        self.states = self.states.unsqueeze(-1)
        self.observations = self.observations.unsqueeze(-1)


