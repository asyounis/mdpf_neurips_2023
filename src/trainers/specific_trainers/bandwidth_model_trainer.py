
#Standard Imports 
import numpy as np
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *

# Project Imports
from trainers.trainer import *
from problems.problem_base import *



class ParticleWeightsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory):

        # Load the data from the directory
        self.all_output_particles = torch.load("{}/particles.pt".format(dataset_directory))
        self.all_output_weights = torch.load("{}/particle_weights.pt".format(dataset_directory))
        self.all_states = torch.load("{}/states.pt".format(dataset_directory))

    def __len__(self):
        return self.all_states.shape[0]

    def __getitem__(self, idx):

        # Compute the return dictionary
        return_dict = {}
        return_dict["states"] = self.all_states[idx]
        return_dict["particles"] = self.all_output_particles[idx]
        return_dict["weights"] = self.all_output_weights[idx]

        return return_dict

class BandwidthModelTrainer(Trainer):
    def __init__(self, model, problem, params, save_dir, device):
        super().__init__(params, model, problem, save_dir, device)

        # Extract the KDE params that we will be using to compute this loss fuction
        assert("kde_params" in self.training_params)
        kde_params = self.training_params["kde_params"]

        # Make sure we have a bandwidth predictor
       	assert(self.main_model.outputs_kde())

        # Get the dataset diretory
        assert("dataset_directory" in self.training_params)
        dataset_directory = self.training_params["dataset_directory"]

        # Get the dataset.  Note: This overrides the dataset already created. 
        # This is a big hack but this is a very special scenario so whatever
        self.training_dataset = ParticleWeightsDataset(dataset_directory)
        self.validation_dataset = self.training_dataset

        # Create the dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_dataset, batch_size=self.training_batch_size, shuffle=True, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.validation_dataset, batch_size=self.validation_batch_size, shuffle=False, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True)

    def get_training_type(self):
        return "bandwidth"

    def do_forward_pass(self, data, dataset, epoch):

        # Unpack the data and move to the device
        states = data["states"].to(self.device)
        particles = data["particles"].to(self.device)
        weights = data["weights"].to(self.device)

        # Get info 
        batch_size = states.shape[0]

        # Compute the best bandwidth for the set of particles
        bandwidths = self.main_model.weighted_bandwidth_predictor(particles.detach(), weights=weights.detach())

        # Combine into a new output dict
        output_dict = dict()
        output_dict["particles"] = particles
        output_dict["particle_weights"] = weights
        output_dict["bandwidths"] = bandwidths

        # Compute the loss
        loss = self.loss_function.compute_loss(output_dict, states)

        # Aggregate the loss via mean
        loss = torch.mean(loss)
            
        return loss, batch_size




# class BandwidthModelTrainer(Trainer):
#     def __init__(self, model, problem, params, save_dir, device):
#         super().__init__(params, model, problem, save_dir, device)

#         # Extract the training parameters
#         self.number_of_particles = self.training_params["number_of_particles"]

#         # Extract the target bandwidths and convert to a tensor for later use
#         self.target_bandwidths = self.training_params["target_bandwidths"]
#         self.target_bandwidths = torch.FloatTensor(self.target_bandwidths)

#         # Extract the KDE params that we will be using to compute this loss fuction
#         assert("kde_params" in self.training_params)
#         kde_params = self.training_params["kde_params"]

#         # Extract what distribution each dim is using
#         self.distribution_types = []
#         for d in kde_params["dims"]:
#             dim_params = kde_params["dims"][d]

#             # create the correct dist for this dim
#             distribution_type = dim_params["distribution_type"]
#             self.distribution_types.append(distribution_type)

#         # Make sure we have a bandwidth predictor
#         assert(self.main_model.outputs_kde())

#         # Get the dataset diretory
#         assert("dataset_directory" in self.training_params)
#         dataset_directory = self.training_params["dataset_directory"]


#         # Get the dataset.  Note: This overrides the dataset already created. 
#         # This is a big hack but this is a very special scenario so whatever
#         self.training_dataset = ParticleWeightsDataset(dataset_directory)
#         self.validation_dataset = self.training_dataset

#         # Create the dataloaders
#         self.train_loader = torch.utils.data.DataLoader(dataset=self.training_dataset, batch_size=self.training_batch_size, shuffle=True, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True)
#         self.validation_loader = torch.utils.data.DataLoader(dataset=self.validation_dataset, batch_size=self.validation_batch_size, shuffle=False, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True)

#     def get_training_type(self):
#         return "bandwidth"

#     def do_forward_pass(self, data, dataset, epoch):

#         # Unpack the data and move to the device
#         states = data["states"].to(self.device)

#         # Get info 
#         batch_size = states.shape[0]

#         # Create the particle and weights set
#         particles, weights = self.create_particle_set_from_states(states)

#         # Compute the best bandwidth for the set of particles
#         bandwidths = self.main_model.weighted_bandwidth_predictor(particles.detach(), weights=weights.detach())

#         # Move the target bands to the correct device
#         if(self.target_bandwidths.device != states.device):
#             self.target_bandwidths = self.target_bandwidths.to(states.device)

#         # Compute the MSE between the desired bandwidth and the output bandwidths
#         loss = (bandwidths - self.target_bandwidths.unsqueeze(0))**2

#         # Aggregate the loss via mean
#         loss = torch.mean(loss)
            
#         return loss, batch_size

#     def create_particle_set_from_states(self, states):

#         #######################################################
#         ## Create the particles
#         #######################################################

#         particles = torch.zeros((states.shape[0], self.number_of_particles, len(self.distribution_types)), device=states.device)

#         for d, dist_type in enumerate(self.distribution_types):
                
#             if(d < states.shape[-1]):
#                 loc = states[:, 0, d]
#             else:
#                 loc = np.random.uniform(low=-10, high=10.0, size=(states.shape[0], ))
#                 loc = torch.FloatTensor(loc).to(states.device)

#             if(dist_type == "Normal"):

#                 bandwidth = np.random.uniform(low=0.01, high=5.0)
#                 dist = D.Normal(loc=loc, scale=bandwidth)
            
#             elif(dist_type == "Von_Mises"):

#                 bandwidth = np.random.uniform(low=0.01, high=5.0)
#                 bandwidth = 1.0 / bandwidth
#                 dist = D.VonMises(loc=loc, concentration=bandwidth)
            
#             else:
#                 assert(False)


#             samples = dist.sample((self.number_of_particles,))
#             samples = torch.permute(samples, (1, 0))
#             particles[..., d] = samples
        
#         #######################################################
#         ## Create the Weights
#         #######################################################

#         weights = torch.rand((states.shape[0], self.number_of_particles), device=states.device)

#         # Normalize the weights
#         norm = torch.sum(weights, dim=-1, keepdim=True)
#         weights = weights / norm

#         return particles, weights















