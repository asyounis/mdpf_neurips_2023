# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


# Project Imports
from models.particle_transformer import *

# Other Models
from models.kde_particle_filter.kde_particle_filter import *


class DiscreteSamplingComparisonModelBase(KDEParticleFilter):

    def __init__(self, model_params, model_architecture_params):
        super(DiscreteSamplingComparisonModelBase, self).__init__(model_params, model_architecture_params)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]

        if("use_resampling_method_always_override" in main_model_arch_params):
            self.use_resampling_method_always_override = main_model_arch_params["use_resampling_method_always_override"]
        else:
            self.use_resampling_method_always_override = False

    def resample_particles(self, input_dict):
        ''' Re-sample the particles to generate a new set of particles.
            This is done via sampling with replacement
        '''

        # Unpack the input
        do_resample = input_dict["do_resample"]
        particles = input_dict["particles"]


        # Select the weights on if we are decoupled or not
        if(self.decouple_weights_for_resampling):
            particle_weights = input_dict["resampling_particle_weights"]
        else:
            particle_weights = input_dict["particle_weights"]

        # Extract information about the input
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]
        device = particles.device

        if(not do_resample):

            # If there are no bandwidths then we just pass the particles through without any changes
            resampled_particles = particles
            resampled_particle_weights = particle_weights

        elif((self.training == False) and (self.use_resampling_method_always_override == False)):


	        selected = None
	        with torch.no_grad():

	            # Compute the cumulative sum of the particle weights so we can use uniform sampling
	            particle_weights_cumsum = torch.cumsum(particle_weights, dim=-1)
	            particle_weights_cumsum = torch.tile(particle_weights_cumsum.unsqueeze(1), [1, number_of_particles, 1])

	            # Generate random numbers, use the same random numbers for all batches
	            uniform_random_nums = torch.rand(size=(batch_size, number_of_particles, 1),device=device)

	            # Select the particle indices's
	            selected = particle_weights_cumsum >= uniform_random_nums
	            _, selected = torch.max(selected, dim=-1)

	            # Resample
	            resampled_particles = torch.zeros(size=(batch_size, number_of_particles, particles.shape[-1]), device=device)
	            for b in range(batch_size):
	                resampled_particles[b,...] = particles[b,selected[b] ,...]
	            resampled_particles = torch.cat([particles[b,selected[b] ,...].unsqueeze(0) for b in range(batch_size)])

	        # Generate the new weights which is all ones since we just re-sampled
	        resampled_particle_weights = torch.full(size=(batch_size, number_of_particles), fill_value=(1/float(number_of_particles)), device=device)
            
        else:
        	resampled_particles,resampled_particle_weights =  self.specific_resampling_method(input_dict)

        # Norm the particles.  Most of the time this wont do anything but sometimes we want to  norm some dims
        resampled_particles = self.particle_transformer.apply_norm(resampled_particles)


        return resampled_particles, resampled_particle_weights, torch.ones(size=(batch_size, 1), device=particles.device)


    def specific_resampling_method(self, input_dict):
    	raise NotImplemented