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
from models.optimal_transport_df.resamplers import *
from models.kde_particle_filter.kde_particle_filter import *
from models.comparison_model_base import *

# class OptimalTransportParticleFilter(KDEParticleFilter):

#     def __init__(self, model_params, model_architecture_params):
#         super(OptimalTransportParticleFilter, self).__init__(model_params, model_architecture_params)

#         # Delete the models that we dont want to use here
#         self.weighted_bandwidth_predictor = None
#         self.resampling_weighted_bandwidth_predictor = None
#         self.resampling_weight_model = None

#         # Specify the OT resampler
#         self.resampler = OTResampler()

#     def resample_particles(self, input_dict):
#         ''' Re-sample the particles to generate a new set of particles.
#             This is done via sampling with replacement
#         '''

#         # Unpack the input
#         particles = input_dict["particles"]

#         # Extract information about the input
#         batch_size = particles.shape[0]
#         number_of_particles = particles.shape[1]
#         device = particles.device

#         # Select the weights on if we are decoupled or not
#         if(self.decouple_weights_for_resampling):
#             particle_weights = input_dict["resampling_particle_weights"]
#         else:
#             particle_weights = input_dict["particle_weights"]

#         # Normalize the particle weights
#         particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)

#         # Sample
#         particles = self.particle_transformer.backward_tranform(particles)
#         resampled_particles, resampled_particle_weights, _  = self.resampler(particles, particle_weights)
#         resampled_particles = self.particle_transformer.forward_tranform(resampled_particles)

#         # Norm the particles.  Most of the time this wont do anything but sometimes we want to  norm some dims
#         resampled_particles = self.particle_transformer.apply_norm(resampled_particles)

#         return resampled_particles, resampled_particle_weights, torch.ones(size=(batch_size, 1), device=particles.device)

#     def outputs_kde(self):
#         return False

#     def outputs_particles_and_weights(self):
#         return True

#     def outputs_single_solution(self):
#         return False



class OptimalTransportParticleFilterLearnedBand(DiscreteSamplingComparisonModelBase):

    def __init__(self, model_params, model_architecture_params):
        super(OptimalTransportParticleFilterLearnedBand, self).__init__(model_params, model_architecture_params)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]

        # Delete the models that we dont want to use here
        self.resampling_weighted_bandwidth_predictor = None
        self.resampling_weight_model = None

        # Get the resampling mix parameter
        if("temperature" not in main_model_arch_params):
            self.temperature = 0.01
        else:
            self.temperature = main_model_arch_params["temperature"]


        # Specify the OT resampler
        self.resampler = OTResampler(epsilon=self.temperature)

    def specific_resampling_method(self, input_dict):
        ''' Re-sample the particles to generate a new set of particles.
            This is done via sampling with replacement
        '''

        # Unpack the input
        particles = input_dict["particles"]

        # Extract information about the input
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]
        device = particles.device

        # Select the weights on if we are decoupled or not
        if(self.decouple_weights_for_resampling):
            particle_weights = input_dict["resampling_particle_weights"]
        else:
            particle_weights = input_dict["particle_weights"]

        # Normalize the particle weights
        particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)

        # Sample
        particles = self.particle_transformer.backward_tranform(particles)
        resampled_particles, resampled_particle_weights, _  = self.resampler(particles, particle_weights)
        resampled_particles = self.particle_transformer.forward_tranform(resampled_particles)

        # Norm the particles.  Most of the time this wont do anything but sometimes we want to  norm some dims
        resampled_particles = self.particle_transformer.apply_norm(resampled_particles)

        if(torch.sum(torch.isnan(resampled_particles)) > 0):
            print("")
            print("")
            print("resampled_particles NAN")

            print("")

            print("resampled_particles", torch.sum(torch.isnan(resampled_particles)))
            print("resampled_particle_weights", torch.sum(torch.isnan(resampled_particle_weights)))
            print("particles", torch.sum(torch.isnan(particles)))
            print("particle_weights", torch.sum(torch.isnan(particle_weights)))
            print("particle_weights log", torch.sum(torch.isnan(particle_weights.log())))
            print("")


            exit(0)

        if(torch.sum(torch.isnan(resampled_particle_weights)) > 0):
            print("resampled_particle_weights NAN")
            exit(0)


        return resampled_particles, resampled_particle_weights

    def outputs_kde(self):
        return True

    def outputs_particles_and_weights(self):
        return True

    def outputs_single_solution(self):
        return False





# class OptimalTransportParticleFilterROTBandwidth(OptimalTransportParticleFilter):

#     def __init__(self, model_params, model_architecture_params):
#         super(OptimalTransportParticleFilterROTBandwidth, self).__init__(model_params, model_architecture_params)

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


