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



class ConcreteParticleFilterLearnedBand(DiscreteSamplingComparisonModelBase):

    def __init__(self, model_params, model_architecture_params):
        super(ConcreteParticleFilterLearnedBand, self).__init__(model_params, model_architecture_params)

        # Delete the models that we dont want to use here
        self.resampling_weighted_bandwidth_predictor = None
        self.resampling_weight_model = None

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]

        # Get the resampling mix parameter
        assert("concrete_relaxation_temperature_parameter" in main_model_arch_params)
        self.concrete_relaxation_temperature_parameter = main_model_arch_params["concrete_relaxation_temperature_parameter"]


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


        # Create the gumbel samples
        log_weights = torch.log(particle_weights + 1e-12).unsqueeze(1)
        log_weights = torch.tile(log_weights, [1, number_of_particles, 1])
        gumbel_samples = torch.nn.functional.gumbel_softmax(log_weights)
        gumbel_samples = gumbel_samples.unsqueeze(-1) 

        # Get the resampled particles
        resampled_particles = particles.unsqueeze(1) * gumbel_samples
        resampled_particles = torch.sum(resampled_particles, dim=2)


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

