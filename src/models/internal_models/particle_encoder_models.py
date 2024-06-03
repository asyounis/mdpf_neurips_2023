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
from models.internal_models.observation_encoder_models import *

class ParticleEncoderBase(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(ParticleEncoderBase, self).__init__(model_parameters)

        # We need the particle dims if we are going to ignore things
        # Also we need it in general
        assert("particle_dim" in model_parameters)
        self.particle_dim = model_parameters["particle_dim"]

        # See if we have ignore dims
        if("particle_ignore_dims" in model_parameters):
            self.particle_ignore_dims = model_parameters["particle_ignore_dims"]
            self.particle_select_dims_cache = None

            # Compute the encoder input particle dime
            self.encoder_input_particle_dim = self.particle_dim - len(self.particle_ignore_dims)

        else:
            self.particle_ignore_dims = None

            # Compute the encoder input particle dime
            self.encoder_input_particle_dim = self.particle_dim


    def create_encoder(self, model_parameters, input_dim_addition):

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_layers" in model_parameters)
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed for the encoder
        encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_layers = model_parameters["encoder_number_of_layers"]
        non_linear_type = model_parameters["non_linear_type"]

        if("encoder_use_batch_norm" in model_parameters):
            encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        else:
            encoder_use_batch_norm = "None"

        if("encoder_use_layer_norm" in model_parameters):
            encoder_use_layer_norm = model_parameters["encoder_use_layer_norm"]
        else:
            encoder_use_layer_norm = "None"



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
        assert(encoder_number_of_layers >= 1)

        # Create the encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=self.encoder_input_particle_dim+input_dim_addition,out_features=encoder_latent_space)))
        
        # Enable or disable the batch norm!
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(encoder_latent_space))

        # Enable or disable the batch norm!
        if(encoder_use_layer_norm == "pre_activation"):
            layers.append(nn.LayerNorm(encoder_latent_space))

        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_layers-1):
            layers.append(non_linear_object())            

            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))

            if(encoder_use_layer_norm == "post_activation"):
                layers.append(nn.LayerNorm(encoder_latent_space))

            layers.append(self.apply_parameter_norm(nn.Linear(in_features=encoder_latent_space,out_features=encoder_latent_space)))

            # Enable or disable the batch norm!
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))

            if(encoder_use_layer_norm == "pre_activation"):
                layers.append(nn.LayerNorm(encoder_latent_space))

        # Generate the model
        self.encoder = nn.Sequential(*layers)


    def process_particle_ignore_dims(self, particles):

        # Nothing to ignore
        if(self.particle_ignore_dims is None):
            return particles

        # Do this once, figure out which dims we want to keep
        if(self.particle_select_dims_cache is None):
            keep_dims = []
            for d in range(self.particle_dim):
                if(d not in self.particle_ignore_dims):
                    keep_dims.append(d)
            assert(len(keep_dims) != 0)

            self.particle_select_dims_cache = torch.tensor(keep_dims).to(particles.device)

        # Keep the dims we want and igore all the rest
        x = torch.index_select(particles, -1, self.particle_select_dims_cache)
        return x 


class ParticleEncoder(ParticleEncoderBase):
    def __init__(self, model_parameters):
        super(ParticleEncoder, self).__init__(model_parameters)

        # Create the encoder
        self.create_encoder(model_parameters, 0)

    def forward(self, particles):

        # Ignore certain dims if needed 
        x = self.process_particle_ignore_dims(particles)

        # Flatten the particles
        x = x.reshape(-1, x.shape[-1])

        # Do the network
        x = self.encoder(x)

        # reshape back into the correct unflattened shape
        x = x.reshape(particles.shape[0], particles.shape[1], -1)

        return x


class ParticleEncoderNoisy(ParticleEncoderBase):
    def __init__(self, model_parameters):
        super(ParticleEncoderNoisy, self).__init__(model_parameters)

        # Create the encoder
        self.create_encoder(model_parameters, 0)

        if("noise_scale_factor" in model_parameters):
            self.noise_scale_factor = model_parameters["noise_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.noise_scale_factor, list)):
                self.noise_scale_factor = torch.FloatTensor(self.noise_scale_factor)
                self.noise_scale_factor = self.noise_scale_factor.unsqueeze(0)

        else:
            self.noise_scale_factor = 1.0

    def forward(self, particles):

        # Ignore certain dims if needed 
        x = self.process_particle_ignore_dims(particles)

        # Flatten the particles
        x = x.reshape(-1, x.shape[-1])

        # Generate Noise for this action
        random_noise_dist = D.Normal(torch.as_tensor(0.0,device=x.device), torch.as_tensor(1.0, device=x.device))
        noise = random_noise_dist.sample(x.shape)

        # Scale the noise
        if(torch.is_tensor(self.noise_scale_factor) and (self.noise_scale_factor.device != x.device)):
            self.noise_scale_factor = self.noise_scale_factor.to(x.device)
        noise = noise * self.noise_scale_factor

        # Add Noise to the particles
        x = x + noise

        # Do the network
        x = self.encoder(x)

        # reshape back into the correct unflattened shape
        x = x.reshape(particles.shape[0], particles.shape[1], -1)

        return x






class House3DMapParticleEncoder(LearnedInternalModelBase):

    def __init__(self, model_parameters):
        super(House3DMapParticleEncoder, self).__init__(model_parameters)

        # Make sure we have the correct parameters are passed in
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed
        self.non_linear_type = model_parameters["non_linear_type"]


        # Create the initial conv layers
        self.conv_layers_1 = nn.ModuleList()
        self.conv_layers_1.append(self.make_conv_layer(1, 24, (3, 3), dilation=1))
        self.conv_layers_1.append(self.make_conv_layer(1, 16, (5, 5), dilation=1))
        self.conv_layers_1.append(self.make_conv_layer(1, 8, (7, 7), dilation=1))
        self.conv_layers_1.append(self.make_conv_layer(1, 8, (7, 7), dilation=2))
        self.conv_layers_1.append(self.make_conv_layer(1, 8, (7, 7), dilation=3))

        # The max pooling layer
        self.max_pooling_layer = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        #  The second conv layer
        self.conv_layers_2 = nn.ModuleList()
        self.conv_layers_2.append(self.make_conv_layer(64, 4, (3, 3), dilation=1, use_activation=False))
        self.conv_layers_2.append(self.make_conv_layer(64, 4, (5, 5), dilation=1, use_activation=False))

        # The activations
        self.activation_1 = self.get_activation_object()

    def make_conv_layer(self, in_channels, out_channels, kernel_size, dilation, padding="same", use_activation=True):

        layers = []

        # Make the convolution layer
        layers.append(self.apply_parameter_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)))

        # Add the non_linearity
        if(use_activation):
            layers.append(self.get_activation_object())

        # Generate the layers
        return nn.Sequential(*layers)

    def get_activation_object(self):
        # Select the non_linear type object to use
        if(self.non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(self.non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(self.non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(self.non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        return non_linear_object()

    def forward(self, x):

        # The first conv layers
        conv_layers_1_outs = []
        for cl in self.conv_layers_1:
            conv_layers_1_outs.append(cl(x))

        # Concat the outputs
        x = torch.cat(conv_layers_1_outs, dim=1)

        # Max pool!!!
        x = self.max_pooling_layer(x)

        # The second conv layers
        conv_layers_2_outs = []
        for cl in self.conv_layers_2:
            conv_layers_2_outs.append(cl(x))

        # Final cat
        x = torch.cat(conv_layers_2_outs, dim=1)

        x = self.activation_1(x)

        return x






def create_particle_encoder_model(model_name, model_parameters):

    model_type = model_parameters[model_name]["type"]

    if(model_type == "ParticleEncoder"):
        return ParticleEncoder(model_parameters[model_name])
    elif(model_type == "ParticleEncoderNoisy"):
        return ParticleEncoderNoisy(model_parameters[model_name])
    elif(model_type == "ImageObservationEncoder"):
        return ImageObservationEncoder(model_parameters[model_name])
    elif(model_type == "ImageObservationWithPoolingEncoder"):
        return ImageObservationWithPoolingEncoder(model_parameters[model_name])
    elif(model_type == "House3DMapParticleEncoder"):
        return House3DMapParticleEncoder(model_parameters[model_name])

    else:
            print("Unknown Model type in ProposalParticleEncoderModelTrainer")
            exit()
