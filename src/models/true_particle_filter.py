# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# For drawing the compute graphs
import torchviz

# Project Imports
from utils import *

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation import kernel_density_estimation as kde



class TruePF(nn.Module):

    def __init__(self):
        super(TruePF, self).__init__()


        # The dim is 1
        self.particle_dimensions = 1

    def resample_particles(self, particles, particle_weights):
        ''' Re-sample the particles to generate a new set of particles.
            The particles are resampled with replacement using the weight of the particles.
            This method assumes the particles are not a KDE

    
            Arguments:
                particles: The particles used for resampling
                particle_weights: The particle weights used for resampling
        '''

        # Get the device the particles are on
        device = particles.device

        # Extract information about the input
        batch_size = particles.shape[0]

        # Keep the number of particles pre and pose sampling the same
        number_of_particles = particles.shape[1]

        # Compute the cumulative sum of the particle weights so we can use uniform sampling
        particle_weights_cumsum = torch.cumsum(particle_weights, dim=-1)
        particle_weights_cumsum = torch.tile(particle_weights_cumsum.unsqueeze(1), [1, number_of_particles, 1])

        # Generate random numbers, use the same random numbers for all batches
        uniform_random_nums = torch.rand(size=(batch_size, number_of_particles, 1)).to(device)

        # Select the particle indices's
        selected = particle_weights_cumsum >= uniform_random_nums
        _, selected = torch.max(selected, dim=-1)


        # If we are not in KDE mode then do a standard resample with replacement
        resampled_particles = torch.zeros(size=(batch_size, number_of_particles, particles.shape[-1])).to(device)
        for b in range(batch_size):
            resampled_particles[b,...] = particles[b,selected[b] ,...]

        # Generate the new weights which is all ones since we just re-sampled
        resampled_particle_weights = torch.ones(size=(batch_size, number_of_particles)).to(device)
        resampled_particle_weights /= float(number_of_particles)

        return resampled_particles, resampled_particle_weights

    def forward(self, particles, particle_weights, observation, timestep_number):


        # Makes sure that the input is just 1 step in sequence and not a the whole sequence
        assert(len(particles.shape) == 3)

        # Extract information about the input
        batch_size = particles.shape[0]

        ############################################################################################# 
        ## Step 1: Re-sample the particles
        ############################################################################################# 
        new_paticles, new_particle_weights = self.resample_particles(particles, particle_weights)

        ############################################################################################ 
        # Step 2: Augment the particles
        ############################################################################################ 
        old_particles = particles.clone()

        # Propagate according to the true dynamics
        new_paticle_mean = particles / 2.0
        new_paticle_mean = new_paticle_mean + (25.0 * (particles / ((particles**2) + 1)))
        new_paticle_mean = new_paticle_mean + (8.0 * np.cos(1.2*timestep_number))
        
        random_dist = D.Normal(0,np.sqrt(10))
        particles = new_paticle_mean + random_dist.rsample(new_paticle_mean.shape).cuda()

        ############################################################################################# 
        ## Step 3: Weight the particles based on the observation
        ############################################################################################# 

        obs_mean = (particles**2) / 20.0
        random_obs = D.Normal(obs_mean.squeeze(-1), np.sqrt(1))
        obs_weights = random_obs.log_prob(observation)
        obs_weights = torch.exp(obs_weights)

        particle_weights = particle_weights * obs_weights
        weight_norms = torch.sum(particle_weights, dim=1)
        particle_weights = particle_weights / weight_norms.unsqueeze(1)


        return particles, particle_weights






