# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# Project Imports
from models.internal_models.internal_model_base import *

class InitializerModelBound(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(InitializerModelBound, self).__init__(model_parameters)

        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_model(model_parameters)

    def create_model(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("input_observation_encoding_dimension" in model_parameters)
        assert("latent_space" in model_parameters)
        assert("number_of_layers" in model_parameters)
        assert("encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)
        assert("output_dim" in model_parameters)
        assert("mins" in model_parameters)
        assert("maxs" in model_parameters)

        # Extract the parameters needed or the timestep encoder
        self.input_observation_encoding_dimension = model_parameters["input_observation_encoding_dimension"]
        latent_space = model_parameters["latent_space"]
        number_of_layers = model_parameters["number_of_layers"]
        encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]
        output_dim = model_parameters["output_dim"]
        self.mins = torch.as_tensor(model_parameters["mins"])
        self.maxs = torch.as_tensor(model_parameters["maxs"])

        # Compute the middle and scaling
        self.output_scaling = (self.maxs - self.mins)/2.0
        self.output_middle = (self.maxs + self.mins)/2.0
        self.output_scaling = self.output_scaling.unsqueeze(0).unsqueeze(0)
        self.output_middle = self.output_middle.unsqueeze(0).unsqueeze(0)


        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        # Need at least 2 layers, the input and output layers
        assert(number_of_layers >= 2)

        # Create the timestamp encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=(self.input_observation_encoding_dimension*2),out_features=latent_space)))
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(latent_space))
        layers.append(non_linear_object())
        if(encoder_use_batch_norm == "post_activation"):
            layers.append(nn.BatchNorm1d(latent_space))

        # the middle layers are all the same fully connected layers
        for i in range(number_of_layers-2):
            layers.append(self.apply_parameter_norm(nn.Linear(in_features=latent_space,out_features=latent_space)))
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(latent_space))

            layers.append(non_linear_object())
            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(latent_space))
        
        # Final layer is the output space and so does not need a non-linearity
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=latent_space, out_features=output_dim)))
        # if(encoder_use_batch_norm):
            # layers.append(nn.BatchNorm1d(output_dim))

        self.final_tanh = nn.Tanh()

        # Generate the model
        self.model = nn.Sequential(*layers)

    def forward(self, encoded_observation, number_of_particles, noise=None):

        # Create an encoding per particle
        encoded_observation_tiled = torch.tile(encoded_observation.unsqueeze(1), [1,number_of_particles,1])

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_observation.device), torch.as_tensor(1.0, device=encoded_observation.device))
            noise = self.random_noise_dist.sample((encoded_observation.shape[0], number_of_particles, self.input_observation_encoding_dimension))
            noise = noise.to(encoded_observation.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_observation_tiled, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_observation.shape[0], number_of_particles, -1)

        # Do the final output and scale
        out = self.final_tanh(out)
        out = (out * self.output_scaling.to(out.device)) + self.output_middle.to(out.device)

        # Return the particles
        return out

def create_initializer_model(model_name, model_parameters):

    # Extract the model type
    model_type = model_parameters[model_name]["type"]

    if(model_type == "InitializerModelBound"):
        parameters = model_parameters[model_name]
        return  InitializerModelBound(parameters)

    else:
        print("Unknown initializer_model type \"{}\"".format(model_type))
        exit()



