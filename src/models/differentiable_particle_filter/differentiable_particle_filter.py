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

# For drawing the compute graphs
import torchviz

# Project Imports
from models.particle_transformer import *

# Other Models
from models.sequential_models import *
from models.internal_models.proposal_models import *
from models.internal_models.weight_models import *
from models.internal_models.particle_encoder_models import *
from models.internal_models.observation_encoder_models import *
from models.internal_models.initializer_models import *
from models.internal_models.bandwidth_models import *
from models.kde_particle_filter.kde_particle_filter import *
from models.comparison_model_base import *

# class DifferentiableParticleFilter(DiscreteSamplingComparisonModelBase):

#     def __init__(self, model_params, model_architecture_params):
#         super(DifferentiableParticleFilter, self).__init__(model_params, model_architecture_params)

#         # Delete the models that we dont want to use here
#         self.weighted_bandwidth_predictor = None
#         self.resampling_weighted_bandwidth_predictor = None
#         self.resampling_weight_model = None

#     def resample_particles(self, input_dict):
#         ''' Re-sample the particles to generate a new set of particles.
#             This is done via sampling with replacement
#         '''

#         # Unpack the input
#         particles = input_dict["particles"]
#         do_resample = input_dict["do_resample"]


#         if(not do_resample):

#             # If there are no bandwidths then we just pass the particles through without any changes
#             resampled_particles = particles
#             resampled_particle_weights = particle_weights

#         else:

#             # Select the weights on if we are decoupled or not
#             if(self.decouple_weights_for_resampling):
#                 particle_weights = input_dict["resampling_particle_weights"]
#             else:
#                 particle_weights = input_dict["particle_weights"]

#             # Extract information about the input
#             batch_size = particles.shape[0]
#             number_of_particles = particles.shape[1]
#             device = particles.device

#             selected = None
#             with torch.no_grad():

#                 # Compute the cumulative sum of the particle weights so we can use uniform sampling
#                 particle_weights_cumsum = torch.cumsum(particle_weights, dim=-1)
#                 particle_weights_cumsum = torch.tile(particle_weights_cumsum.unsqueeze(1), [1, number_of_particles, 1])

#                 # Generate random numbers, use the same random numbers for all batches
#                 uniform_random_nums = torch.rand(size=(batch_size, number_of_particles, 1),device=device)

#                 # Select the particle indices's
#                 selected = particle_weights_cumsum >= uniform_random_nums
#                 _, selected = torch.max(selected, dim=-1)

#                 # Resample
#                 resampled_particles = torch.zeros(size=(batch_size, number_of_particles, particles.shape[-1]), device=device)
#                 for b in range(batch_size):
#                     resampled_particles[b,...] = particles[b,selected[b] ,...]
#                 resampled_particles = torch.cat([particles[b,selected[b] ,...].unsqueeze(0) for b in range(batch_size)])

#             # Generate the new weights which is all ones since we just re-sampled
#             resampled_particle_weights = torch.full(size=(batch_size, number_of_particles), fill_value=(1/float(number_of_particles)), device=device)

#         # Norm the particles.  Most of the time this wont do anything but sometimes we want to  norm some dims
#         resampled_particles = self.particle_transformer.apply_norm(resampled_particles)

#         return resampled_particles, resampled_particle_weights, torch.ones(size=(batch_size, 1), device=particles.device)

#     def outputs_kde(self):
#         return False

#     def outputs_particles_and_weights(self):
#         return True

#     def outputs_single_solution(self):
#         return False


class DifferentiableParticleFilterLearnedBand(DiscreteSamplingComparisonModelBase):

    def __init__(self, model_params, model_architecture_params):
        super(DifferentiableParticleFilterLearnedBand, self).__init__(model_params, model_architecture_params)

        # Delete the models that we dont want to use here
        self.resampling_weighted_bandwidth_predictor = None
        self.resampling_weight_model = None

    def specific_resampling_method(self, input_dict):
        ''' Re-sample the particles to generate a new set of particles.
            This is done via sampling with replacement
        '''

        # Unpack the input
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

        # Norm the particles.  Most of the time this wont do anything but sometimes we want to  norm some dims
        resampled_particles = self.particle_transformer.apply_norm(resampled_particles)

        return resampled_particles, resampled_particle_weights

    def outputs_kde(self):
        return True

    def outputs_particles_and_weights(self):
        return True

    def outputs_single_solution(self):
        return False



# class DifferentiableParticleFilterROTBandwidth(DifferentiableParticleFilter):

#     def __init__(self, model_params, model_architecture_params):
#         super(DifferentiableParticleFilterROTBandwidth, self).__init__(model_params, model_architecture_params)

#         # Delete the models that we dont want to use here
#         self.resampling_weighted_bandwidth_predictor = None
#         self.resampling_weight_model = None

#         # Extract the name of the model so we can get its model architecture params
#         main_model_name = model_params["main_model"]
#         main_model_arch_params = model_architecture_params[main_model_name]

#         # Make this the rule of thumb predictor
#         self.weighted_bandwidth_predictor = RuleOfThumbBandwidth(main_model_arch_params)

#     def outputs_kde(self):
#         return True

#     def outputs_particles_and_weights(self):
#         return True

#     def outputs_single_solution(self):
#         return False


