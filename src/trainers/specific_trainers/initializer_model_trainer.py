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


class InitializerModelTrainer(Trainer):
    def __init__(self, model, problem, params, save_dir, device):
        super().__init__(params, model, problem, save_dir, device)
        
        # Extract the learning rates 
        self.number_of_particles = self.training_params["number_of_particles"]

    def get_training_type(self):
        return "initilizer"


    def do_forward_pass(self, data, dataset, epoch):

        # Unpack the data and move to the device
        observations = data["observations"].to(self.device)
        states = data["states"].to(self.device)


        observation_is_image = False
        if(len(observations.shape) == 5):

            # If the observation has 5 dims ([batch, seq, channels, width, height]) then it is an image
            observation_is_image = True

        # Flatten all the observations and states since we dont care about sequences here
        # we are going to look at each obs/state pair individually
        if(observation_is_image):
            observations = torch.reshape(observations, (-1,observations.shape[-3], observations.shape[-2], observations.shape[-1]))
        else:
            observations = torch.reshape(observations, (-1, observations.shape[-1]))
        states = torch.reshape(states, (-1, states.shape[-1]))

        # Transform the Observations
        observations = self.problem.observation_transformer.forward_tranform(observations)

        # Extract some statistics
        number_of_samples = observations.shape[0]

        # Encode the observation 
        encoded_observation = self.main_model.observation_encoder(observations)

        # Predict a particle set
        new_particles = self.main_model.initializer_model(encoded_observation, self.number_of_particles)

        # Decode the particle 
        new_particles = self.main_model.particle_transformer.forward_tranform(new_particles)

        # Assume equally weighted particles for the KDE
        new_particle_weights = torch.ones(size=(number_of_samples, self.number_of_particles), device=self.device)
        new_particle_weights = new_particle_weights / self.number_of_particles


        # Create the "output dict" so that we can feed it into the loss function
        output_dict = dict()
        output_dict["particles"] = new_particles
        output_dict["particle_weights"] = new_particle_weights

        # If the model can output a bandwidth then lets get it
        if(self.main_model.outputs_kde()):

            # Compute the best bandwidth for the set of particles
            bandwidths = self.main_model.weighted_bandwidth_predictor(new_particles.detach(), weights=new_particle_weights.detach())
            output_dict["bandwidths"] = bandwidths

        # Compute the loss           
        loss = self.loss_function.compute_loss(output_dict, states)

        # Aggregate the loss via mean
        loss = torch.mean(loss)

        return loss, number_of_samples











