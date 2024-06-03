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
import functools

# Project Imports
from models.internal_models.internal_model_base import *
from models.internal_models.custom_layers import *


##################################################################################################################################
## Standard Particle
##################################################################################################################################

class ParticleWeigherUnscaledInputProcessor:
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):

        # Add the observation encoder 
        self.observation_encoder = observation_encoder

        # Add the observation encoder  
        if(particle_encoder is not None):
            self.particle_encoder = particle_encoder
        else:
            self.particle_encoder = None

        # The particle transformer for this problem.  This is not a neural network
        self.particle_transformer = particle_transformer

        # Extract the number of obs we need
        assert("number_of_observation_inputs" in model_parameters)
        self.number_of_observation_inputs = model_parameters["number_of_observation_inputs"]
        assert((self.number_of_observation_inputs == 1) or (self.number_of_observation_inputs == 2))

        # If we should use the old particles in this model
        if("use_old_particles" in model_parameters):
            self.use_old_particles = model_parameters["use_old_particles"]
        else:
            self.use_old_particles = False


    def process(self, input_dict):

        # Unpack
        observation = input_dict["observation"]
        next_observation = input_dict["next_observation"]
        internal_particles = input_dict["internal_particles"]
        old_particles = input_dict["old_particles"]

        # if we have no observation then there is nothing to weight so no need to prepare any of the inputs
        if(observation is None):
            return None

        # If we need the next observation and we dont have it  then there is nothing to weight so no need to prepare any of the inputs
        if((self.number_of_observation_inputs == 2) and (next_observation is None)):
            return None

        # All the final inputs for the network
        final_weight_net_inputs = []

        # Encode the and add the observations
        encoded_observation = self.observation_encoder(observation)
        final_weight_net_inputs.append(torch.tile(encoded_observation[:,:].unsqueeze(1),[1,internal_particles.shape[1], 1]))

        # Encode the and add the next observations if we need to
        if(self.number_of_observation_inputs == 2):
            encoded_next_observation = self.observation_encoder(next_observation)
            final_weight_net_inputs.append(torch.tile(encoded_next_observation[:,:].unsqueeze(1),[1,internal_particles.shape[1], 1]))

        # Encode (if needed) and add the internal particles
        if(self.particle_encoder is not None):
            final_weight_net_inputs.append(self.particle_encoder(internal_particles))
        else:
            final_weight_net_inputs.append(internal_particles)

        # Encode (if needed) and add the internal old particles if we are going to use them
        if(self.use_old_particles):
            # Need to transform the particles from output space to internal space
            # Note "internal_particles" is already in internal space
            internal_old_particles = self.particle_transformer.backward_tranform(old_particles)

            if(self.particle_encoder is not None):
                final_weight_net_inputs.append(self.particle_encoder(internal_old_particles))
            else:
                final_weight_net_inputs.append(internal_old_particles)

        # The final input is all the encoding parts put into 1 big vector 
        final_weight_net_inputs = torch.cat(final_weight_net_inputs,dim=-1)
        final_weight_net_inputs = final_weight_net_inputs.view(-1, final_weight_net_inputs.shape[-1])

        # Pack into a dict
        return_dict = dict()
        return_dict["final_input"] = final_weight_net_inputs
        return_dict["batch_size"] = internal_particles.shape[0]
        return_dict["number_of_particles"] = internal_particles.shape[1]

        return return_dict


    def is_identical(self, other_processor):
        ''' 
            Check if 2 processors are identical.  We need this since if 2 processor are identical then we dont need to call them twice with the same inputs.
            Just call 1 of them and reuse the output
        '''
        if(self.observation_encoder != other_processor.observation_encoder):
            return False

        if(self.particle_encoder != other_processor.particle_encoder):
            return False

        if(self.particle_transformer != other_processor.particle_transformer):
            return False

        if(self.number_of_observation_inputs != other_processor.number_of_observation_inputs):
            return False

        if(self.use_old_particles != other_processor.use_old_particles):
            return False

        return True

class ParticleWeigherUnscaled(LearnedInternalModelBase):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherUnscaled, self).__init__(model_parameters)

        # Create an input processor for this model
        self.input_processor = ParticleWeigherUnscaledInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Create the encoder model
        self.create_encoder(model_parameters)

    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("input_particle_dimension" in model_parameters)
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_layers" in model_parameters)
        assert("observation_encoder_output" in model_parameters)
        assert("number_of_observation_inputs" in model_parameters)
        assert("use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed or the timestep encoder
        self.input_particle_dimension = model_parameters["input_particle_dimension"]
        encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_layers = model_parameters["encoder_number_of_layers"]
        observation_encoder_output = model_parameters["observation_encoder_output"]
        use_batch_norm = model_parameters["use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]

        # We can only encode either 1 or 2 inputs
        number_of_observation_inputs = model_parameters["number_of_observation_inputs"]
        assert((number_of_observation_inputs == 1) or (number_of_observation_inputs == 2))

        # If we should use the old particles in this model
        if("use_old_particles" in model_parameters):
            use_old_particles = model_parameters["use_old_particles"]
        else:
            use_old_particles = False

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
        assert(encoder_number_of_layers >= 2)

        # Compute the size of the input layer
        input_layer_dim = 0
        input_layer_dim += (observation_encoder_output * number_of_observation_inputs)
        input_layer_dim += self.input_particle_dimension

        if(use_old_particles):
            input_layer_dim += self.input_particle_dimension

        # Create the timestamp encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=input_layer_dim,out_features=encoder_latent_space)))
        if(use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(encoder_latent_space))
        layers.append(non_linear_object())
        if(use_batch_norm == "post_activation"):
            layers.append(nn.BatchNorm1d(encoder_latent_space))
        

        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_layers-2):
            
            layers.append(self.apply_parameter_norm(nn.Linear(in_features=encoder_latent_space,out_features=encoder_latent_space)))
            if(use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))

            layers.append(non_linear_object())
            if(use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))


        # Final layer is the output space and so does not need a non-linearity
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=encoder_latent_space, out_features=1)))
        # if(use_batch_norm):
        #     layers.append(nn.BatchNorm1d(1))

        # Generate the models
        self.encoder = nn.Sequential(*layers)

    def forward(self, input_dict):

        # Unpack
        final_input = input_dict["final_input"]
        batch_size = input_dict["batch_size"]
        number_of_particles = input_dict["number_of_particles"]

        # Do the forward pass
        out = self.encoder(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(batch_size, number_of_particles)

        return out

    def get_number_of_observation_inputs(self):
        return self.number_of_observation_inputs

class ParticleWeigher(ParticleWeigherUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigher, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # We need a sigmoid for this model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)

        # Scale via sigmoid
        out = self.sigmoid(out)

        # Make sure we are at least the min value
        out = out * (1.0 - self.min_obs_likelihood)
        out = out + self.min_obs_likelihood 

        return out

class ParticleUnboundPositive(ParticleWeigherUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleUnboundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # The final softplus activation
        self.softplus = nn.Softplus()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)


        # Make positive via exponential
        out = torch.exp(out)

        # out = self.softplus(out)

        # Make sure we are at least the min value
        out = out + self.min_obs_likelihood 

        return out

class ParticleBoundPositive(ParticleWeigherUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleBoundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # Extract the max value we will allow for the observation
        assert("max_obs_likelihood" in model_parameters)
        self.max_obs_likelihood = model_parameters["max_obs_likelihood"]

        # The final softplus activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)

        out = self.sigmoid(out)
        out = out* (self.max_obs_likelihood - self.min_obs_likelihood)
        out = out + self.min_obs_likelihood 

        return out






##################################################################################################################################
## Dual Encoder
##################################################################################################################################

class ParticleWeigherBoundingBoxInputProcessor:
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):

        # Add the observation encoder 
        self.observation_encoder = observation_encoder

        # Add the observation encoder  
        if(particle_encoder is not None):
            self.particle_encoder = particle_encoder
        else:
            self.particle_encoder = None

        # The particle transformer for this problem.  This is not a neural network
        self.particle_transformer = particle_transformer

        # Make sure the parameters we need are present
        assert("bounding_box_center_x_state_dim_pos" in model_parameters)
        assert("bounding_box_center_y_state_dim_pos" in model_parameters)
        assert("bounding_box_width_state_dim_pos" in model_parameters)
        assert("bounding_box_height_state_dim_pos" in model_parameters)
        assert("state_scale" in model_parameters)

        # Extract the parameters
        self.bounding_box_center_x_state_dim_pos = model_parameters["bounding_box_center_x_state_dim_pos"]
        self.bounding_box_center_y_state_dim_pos = model_parameters["bounding_box_center_y_state_dim_pos"]
        self.bounding_box_width_state_dim_pos = model_parameters["bounding_box_width_state_dim_pos"]
        self.bounding_box_height_state_dim_pos = model_parameters["bounding_box_height_state_dim_pos"]
        self.state_scale = model_parameters["state_scale"]

        # Get the image size we should reshape into.  The reshaped patch will be a square!
        self.patch_image_size = self.observation_encoder.input_image_size

        # Create the grids we will be using to crop and resize the patch into
        ls = torch.linspace(0, 1, self.patch_image_size)
        self.x_grid, self.y_grid = torch.meshgrid(ls, ls, indexing="ij")

        # Create the grids we will be using to resize the reference patch into
        ls = torch.linspace(-1, 1, self.patch_image_size)
        self.x_grid_ref_patches, self.y_grid_ref_patches = torch.meshgrid(ls, ls, indexing="ij")


    def process(self, input_dict):

        # Unpack
        reference_patch = input_dict["reference_patch"]
        observations = input_dict["observation"]
        particles = input_dict["particles"]

        # Get some info
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # We need at least 4 dims to specify the bounding box
        assert(particles.shape[-1] >= 4)

        # if we have no observation then there is nothing to weight so no need to prepare any of the inputs
        if(observations is None):
            return None

        # Extract the patches
        extracted_patches = self.extract_and_resize_patches(particles, observations)

        # Collapse the batch and num particles dims so we can pass it through the observation encoder
        extracted_patches = torch.reshape(extracted_patches, (batch_size*number_of_particles, extracted_patches.shape[2], extracted_patches.shape[3], extracted_patches.shape[4]))        

        # Encode the patches
        encoded_extracted_patches = self.observation_encoder(extracted_patches)        

        # Resize the reference patches to the correct size
        resized_reference_patches = self.resize_reference_patches(reference_patch)

        # Encode the reference patches
        encoded_reference_patches = self.observation_encoder(resized_reference_patches)

        # tile it so we have 1 per particle
        encoded_reference_patches = torch.tile(encoded_reference_patches.unsqueeze(1), (1, number_of_particles, 1))        
        encoded_reference_patches =  torch.reshape(encoded_reference_patches, (batch_size*number_of_particles, encoded_reference_patches.shape[-1]))        

        # All the final inputs for the network
        final_weight_net_inputs = []

        # Encode the and add the observations
        final_weight_net_inputs.append(encoded_reference_patches)
        final_weight_net_inputs.append(encoded_extracted_patches)

        # The final input is all the encoding parts put into 1 big vector 
        final_weight_net_inputs = torch.cat(final_weight_net_inputs,dim=-1)
        final_weight_net_inputs = final_weight_net_inputs.view(-1, final_weight_net_inputs.shape[-1])

        # Pack into a dict
        return_dict = dict()
        return_dict["final_input"] = final_weight_net_inputs
        return_dict["batch_size"] = batch_size
        return_dict["number_of_particles"] = number_of_particles

        return return_dict

    def extract_and_resize_patches(self, particles, observations):

        # extract the some information
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Rescale particles to be between -1 and 1
        particles = particles / self.state_scale

        # Make sure the grids are on the correct device
        if((self.x_grid.device != particles.device) or (self.y_grid.device != particles.device)):
            self.x_grid = self.x_grid.to(particles.device)
            self.y_grid = self.y_grid.to(particles.device)

        # Tile the grids so we have 1 per batch and per particle
        tiled_x_grid = torch.tile(self.x_grid.unsqueeze(0).unsqueeze(0), (particles.shape[0], particles.shape[1], 1, 1))
        tiled_y_grid = torch.tile(self.y_grid.unsqueeze(0).unsqueeze(0), (particles.shape[0], particles.shape[1], 1, 1))

        # extract the widths and heights
        bb_widths = particles[..., self.bounding_box_width_state_dim_pos]
        bb_heights = particles[..., self.bounding_box_height_state_dim_pos]

        # Increase the size
        # bb_widths = bb_widths * 1.5
        # bb_widths = bb_heights * 1.5

        # Compute the sampling locations
        tiled_x_grid = tiled_x_grid * bb_widths.unsqueeze(-1).unsqueeze(-1)
        tiled_y_grid = tiled_y_grid * bb_heights.unsqueeze(-1).unsqueeze(-1)
        tiled_x_grid = tiled_x_grid + (particles[..., self.bounding_box_center_x_state_dim_pos] - (bb_widths / 2)).unsqueeze(-1).unsqueeze(-1)
        tiled_y_grid = tiled_y_grid + (particles[..., self.bounding_box_center_y_state_dim_pos] - (bb_heights / 2)).unsqueeze(-1).unsqueeze(-1)

        # Rescale to be be within [-1, 1]
        # tiled_x_grid = ((tiled_x_grid / observations.shape[-1]) * 2.0) - 1.0
        # tiled_y_grid = ((tiled_y_grid / observations.shape[-2]) * 2.0) - 1.0

        zeros_array = torch.zeros_like(tiled_x_grid)

        grid = torch.stack([tiled_x_grid, tiled_y_grid, zeros_array], dim=-1)

        tiled_observations = observations.unsqueeze(2)

        # Extract and resize the patches all in 1 shot!
        extracted_patches = F.grid_sample(tiled_observations, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
        extracted_patches = torch.permute(extracted_patches, (0, 2, 1, 3, 4))


        return extracted_patches

    def resize_reference_patches(self, reference_patches):

        # extract the some information
        batch_size = reference_patches.shape[0]

        # Make sure the grids are on the correct device
        if((self.x_grid_ref_patches.device != reference_patches.device) or (self.y_grid_ref_patches.device != reference_patches.device)):
            self.x_grid_ref_patches = self.x_grid_ref_patches.to(reference_patches.device)
            self.y_grid_ref_patches = self.y_grid_ref_patches.to(reference_patches.device)

        # Tile the grids so we have 1 per batch and per particle
        tiled_x_grid = torch.tile(self.x_grid_ref_patches.unsqueeze(0), (reference_patches.shape[0], 1, 1))
        tiled_y_grid = torch.tile(self.y_grid_ref_patches.unsqueeze(0), (reference_patches.shape[0], 1, 1))

        grid = torch.stack([tiled_x_grid, tiled_y_grid], dim=-1)

        # Resize the reference patches
        resized_reference_patches = F.grid_sample(reference_patches, grid, align_corners=True, mode="bilinear", padding_mode="zeros")

        img = resized_reference_patches[0]
        img = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()

        return resized_reference_patches


    def is_identical(self, other_processor):
        ''' 
            Check if 2 processors are identical.  We need this since if 2 processor are identical then we dont need to call them twice with the same inputs.
            Just call 1 of them and reuse the output
        '''
        if(self.observation_encoder != other_processor.observation_encoder):
            return False

        if(self.particle_encoder != other_processor.particle_encoder):
            return False

        if(self.particle_transformer != other_processor.particle_transformer):
            return False


        if(self.patch_image_size != other_processor.patch_image_size):
            return False


        if(self.bounding_box_center_x_state_dim_pos != other_processor.bounding_box_center_x_state_dim_pos):
            return False

        if(self.bounding_box_center_y_state_dim_pos != other_processor.bounding_box_center_y_state_dim_pos):
            return False

        if(self.bounding_box_width_state_dim_pos != other_processor.bounding_box_width_state_dim_pos):
            return False

        if(self.bounding_box_height_state_dim_pos != other_processor.bounding_box_height_state_dim_pos):
            return False

        if(self.state_scale != other_processor.state_scale):
            return False

        return True

class ParticleWeigherDualEncoderUnscaled(LearnedInternalModelBase):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherDualEncoderUnscaled, self).__init__(model_parameters)

        # Create the input processor
        self.create_input_processor(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Create the encoder model
        self.create_encoder(model_parameters)

    def create_input_processor(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        # Create an input processor for this model
        self.input_processor = ParticleWeigherBoundingBoxInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)


    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_layers" in model_parameters)
        assert("observation_encoder_output" in model_parameters)
        assert("use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed or the timestep encoder
        encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_layers = model_parameters["encoder_number_of_layers"]
        observation_encoder_output = model_parameters["observation_encoder_output"]
        use_batch_norm = model_parameters["use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]

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
        assert(encoder_number_of_layers >= 2)

        # Compute the size of the input layer. It should be twice the observation encoder output since we 
        # have a dual encoder (aka 2 inputs!)
        input_layer_dim = observation_encoder_output * 2

        # Create the timestamp encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=input_layer_dim,out_features=encoder_latent_space)))
        if(use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(encoder_latent_space))
        layers.append(non_linear_object())
        if(use_batch_norm == "post_activation"):
            layers.append(nn.BatchNorm1d(encoder_latent_space))
        
        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_layers-2):
            
            layers.append(self.apply_parameter_norm(nn.Linear(in_features=encoder_latent_space,out_features=encoder_latent_space)))
            if(use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))

            layers.append(non_linear_object())
            if(use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(encoder_latent_space))


        # Final layer is the output space and so does not need a non-linearity
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=encoder_latent_space, out_features=1)))

        # Generate the models
        self.encoder = nn.Sequential(*layers)

    def forward(self, input_dict):

        # Unpack
        final_input = input_dict["final_input"]
        batch_size = input_dict["batch_size"]
        number_of_particles = input_dict["number_of_particles"]

        # Do the forward pass
        out = self.encoder(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(batch_size, number_of_particles)

        return out

    def get_number_of_observation_inputs(self):
        return 1

class ParticleWeigherDualEncoder(ParticleWeigherDualEncoderUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherDualEncoder, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # We need a sigmoid for this model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)

        # Scale via sigmoid
        out = self.sigmoid(out)

        # Make sure we are at least the min value
        out = out * (1.0 - self.min_obs_likelihood)
        out = out + self.min_obs_likelihood 

        return out

class ParticleWeigherDualEncoderUnboundPositive(ParticleWeigherDualEncoderUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherDualEncoderUnboundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # The final softplus activation
        self.softplus = nn.Softplus()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)


        # Make positive via exponential
        # out = torch.exp(out)

        out = self.softplus(out)

        # Make sure we are at least the min value
        out = out + self.min_obs_likelihood 

        return out

class ParticleWeigherDualEncoderBoundPositive(ParticleWeigherDualEncoderUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherDualEncoderBoundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

        # Extract the min value we will allow for the observation
        assert("min_obs_likelihood" in model_parameters)
        self.min_obs_likelihood = model_parameters["min_obs_likelihood"]

        # Extract the max value we will allow for the observation
        assert("max_obs_likelihood" in model_parameters)
        self.max_obs_likelihood = model_parameters["max_obs_likelihood"]

        # The final softplus activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        # Get the output in unscaled units
        out = super().forward(input_dict)

        out = self.sigmoid(out)
        out = out * (self.max_obs_likelihood - self.min_obs_likelihood)
        out = out + self.min_obs_likelihood 

        return out





##################################################################################################################################
## Affine
##################################################################################################################################

class ParticleWeigherAffineTransformerInputProcessor:
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        
        # Save for later
        self.observation_encoder = observation_encoder
        self.particle_encoder = particle_encoder
        self.particle_transformer = particle_transformer

        # the global map will be down-scaled by some factor (must be a float)
        self.window_scaler = float(8.0)
    
        # The size of the local map (must be an int)
        self.local_map_size = int(28)

    def process(self, input_dict):

        # Unpack
        world_map = input_dict["world_map"]
        observations = input_dict["observation"]
        particles = input_dict["particles"]

        # Get some info
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # We need at least 3 dims to specify the affine transform
        assert(particles.shape[-1] >= 3)

        # if we have no observation then there is nothing to weight so no need to prepare any of the inputs
        if(observations is None):
            return None

        # extract the local maps for each particle
        local_maps = self.extract_local_maps(particles, world_map)

        # Add the channel dims
        local_maps = local_maps.unsqueeze(2)

        # reshape the local maps so that we can encode them 
        local_maps = torch.reshape(local_maps, (batch_size*number_of_particles, local_maps.shape[2], local_maps.shape[3], local_maps.shape[4]))

        # Encode the local maps
        encoded_local_maps = self.particle_encoder(local_maps)

        # Encode the observation
        encoded_obs_raw = self.observation_encoder(observations)

        # tile it so we have 1 per particle
        encoded_obs = torch.tile(encoded_obs_raw.unsqueeze(1), (1, number_of_particles, 1))        
        encoded_obs =  torch.reshape(encoded_obs, (batch_size*number_of_particles, -1))        

        # All the final inputs for the network
        final_weight_net_inputs = []

        # Encode the and add the observations
        final_weight_net_inputs.append(encoded_obs)
        final_weight_net_inputs.append(encoded_local_maps)

        # The final input is all the encoding parts put into 1 big vector 
        final_weight_net_inputs = torch.cat(final_weight_net_inputs,dim=-1)
        final_weight_net_inputs = final_weight_net_inputs.view(-1, final_weight_net_inputs.shape[-1])

        # Pack into a dict
        return_dict = dict()
        return_dict["final_input"] = final_weight_net_inputs
        return_dict["batch_size"] = batch_size
        return_dict["number_of_particles"] = number_of_particles
        # return_dict["local_maps"] = local_maps

        return return_dict


    def extract_local_maps(self, particles, world_map):
        
        # Extract some info
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Flatten the particles down so we can 
        particles = torch.reshape(particles, (batch_size*number_of_particles, -1))

        # Pre-compute sin and cos
        thetas = -particles[...,-1] - (0.5 * np.pi)
        sin_thetas = torch.sin(thetas)
        cos_thetas = torch.cos(thetas)

        # Some other information we need
        map_width = world_map.shape[1]
        map_height = world_map.shape[2]

        # Some useful things to have for later
        one = torch.ones((particles.shape[0], ), device=particles.device)
        zero = torch.zeros((particles.shape[0], ), device=particles.device)

        ##########################################################################################################################################
        ### Create the Affine Matrix
        ##########################################################################################################################################

        # ---------------------------------------------------------------------
        # Step 1: Move the map such that the map is centered around the particle
        # ---------------------------------------------------------------------

        # Compute the translation
        translate_x = particles[..., 0]
        translate_y = particles[..., 1]

        # Scale it since the grid sampler wants it from [-1, 1] not [map_width, map_height]
        map_width_translation, map_height_translation  = self.particle_transformer.downscale_map(float(map_width), float(map_height))
        translate_x = (translate_x * (2.0 / map_width_translation)) - 1.0
        translate_y = (translate_y * (2.0 / map_height_translation)) - 1.0

        # create the matrix
        translation_mat_1 = torch.stack([one, zero, translate_x, zero, one, translate_y, zero, zero, one], dim=-1)
        translation_mat_1 = torch.reshape(translation_mat_1, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 2: Rotate the map such that that the orientation matches that of the state
        # ---------------------------------------------------------------------
        rotation_matrix = torch.stack([cos_thetas, sin_thetas, zero, -sin_thetas, cos_thetas, zero, zero, zero, one], dim=-1)
        rotation_matrix = torch.reshape(rotation_matrix, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 3: scale down the map
        # ---------------------------------------------------------------------
        scale_x = torch.full((particles.shape[0], ), fill_value=(float(self.local_map_size) * self.window_scaler / float(map_width)), device=particles.device)
        scale_y = torch.full((particles.shape[0], ), fill_value=(float(self.local_map_size) * self.window_scaler / float(map_height)), device=particles.device)
        scale_mat = torch.stack([scale_x, zero, zero, zero, scale_y, zero, zero, zero, one], dim =-1)
        scale_mat = torch.reshape(scale_mat, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 4: translate the local map such that the state defines the bottom mid-point instead of the center
        # ---------------------------------------------------------------------
        translate_y = -torch.ones((particles.shape[0] ,), device=particles.device)
        translation_mat_2 = torch.stack([one, zero, zero, zero, one, translate_y, zero, zero, one], dim =-1)
        translation_mat_2 = torch.reshape(translation_mat_2, (-1, 3, 3))

        # ---------------------------------------------------------------------
        # Step 5: Combine all the matrices into a final affine matrix
        # ---------------------------------------------------------------------
        final_affine_matrix = translation_mat_1
        final_affine_matrix = torch.bmm(final_affine_matrix, rotation_matrix)
        final_affine_matrix = torch.bmm(final_affine_matrix, scale_mat)
        final_affine_matrix = torch.bmm(final_affine_matrix, translation_mat_2)
        final_affine_matrix = final_affine_matrix[:, :2, :]


        ########################################################################################################################################
        # Sample the local map using the affine matrix
        ########################################################################################################################################
        
        # Create the affine grid
        affine_sampling_grid = torch.nn.functional.affine_grid(final_affine_matrix, (particles.shape[0], 1, self.local_map_size, self.local_map_size) , align_corners=False)

        # Reshape the grid to have a correct particle dim
        affine_sampling_grid = affine_sampling_grid.view((batch_size, number_of_particles, affine_sampling_grid.shape[1], affine_sampling_grid.shape[2], affine_sampling_grid.shape[3]))

        # Create the world map that has a paritlce dim BUT does not actually copy the data, its just a view to the same data
        world_map = world_map.unsqueeze(1)
        world_map = world_map.expand(-1, number_of_particles, -1, -1)

        # Do our fancy sampling!
        local_maps = self.ali_grid_sampler_2d(world_map, affine_sampling_grid)

        return local_maps

    def is_identical(self, other_processor):
        ''' 
            Check if 2 processors are identical.  We need this since if 2 processor are identical then we dont need to call them twice with the same inputs.
            Just call 1 of them and reuse the output
        '''
        if(self.observation_encoder != other_processor.observation_encoder):
            return False

        if(self.particle_encoder != other_processor.particle_encoder):
            return False

        if(self.particle_transformer != other_processor.particle_transformer):
            return False


        if(self.window_scaler != other_processor.window_scaler):
            return False


        if(self.local_map_size != other_processor.local_map_size):
            return False

        return True

    def ali_grid_sampler_2d(self, a, grid,align_corners=False):

        # zero padding
        padding_mode = 0

        # Bilinear Mode
        interpolation_mode = 0


        def unnormalize(coords, size):
            # Rescale coordinates from [-1, 1] to:
            #   [0, size - 1] if align_corners is True
            #   [-.5, size -.5] if align_corners is False
            mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
            ofs = size * 0.5 - 0.5
            return coords * mul + ofs

        def compute_coordinates(coords, size):
            if padding_mode == 0:  # Zero
                return coords
            
        def compute_source_index(coords, size):
            return unnormalize(coords, size)

        N, P, iH, iW = a.shape
        _, _, oH, oW, _ = grid.shape

        def in_bounds_cond(xs, ys):
            return torch.logical_and(0 <= xs, torch.logical_and(xs < iW, torch.logical_and(0 <= ys, ys < iH)))

        N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
        P_idx = torch.arange(P, device=a.device).view(1, P, 1, 1)

        def clip(xs, ys, ws):
            cond = in_bounds_cond(xs, ys)

            # To clip to inside valid coordinates, we map the coordinates
            # to (x, y) = (0, 0) and also set the weight to 0
            # We also change the shape of the tensor to the appropriate one for
            # broadcasting with N_idx, C_idx for the purposes of advanced indexing
            # return tuple(
            #     torch.where(cond, t, 0).view(N, 1, oH, oW) for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
            # )

            # Huge hack to just get something that is a float...... BOOOO
            zeros = torch.zeros((1,), device=xs.device).float()

            return tuple(
                torch.where(cond, t, zeros).view(N, P, oH, oW) for t in (xs, ys, ws)
            )

        def get_summand(ix, iy, w):
            # Perform clipping, index into input tensor and multiply by weight
            idx_x, idx_y, w_ = clip(ix, iy, w)

            # Convert to ints before indexing
            idx_x = idx_x.to(dtype=torch.int64)
            idx_y = idx_y.to(dtype=torch.int64)

            return a[N_idx, P_idx, idx_y, idx_x] * w_

        # Need this instead of just sum() to keep mypy happy
        def _sum_tensors(ts):
            return functools.reduce(torch.add, ts)


        x = grid[..., 0]
        y = grid[..., 1]


        if interpolation_mode == 0:  # Bilinear
            ix = compute_source_index(x, iW)
            iy = compute_source_index(y, iH)

            ix_nw, iy_nw = ix.floor(), iy.floor()
            ix_ne, iy_ne = ix_nw + 1, iy_nw
            ix_sw, iy_sw = ix_nw, iy_nw + 1
            ix_se, iy_se = ix_ne, iy_sw

            w_nw = (ix_se - ix) * (iy_se - iy)
            w_ne = (ix - ix_sw) * (iy_sw - iy)
            w_sw = (ix_ne - ix) * (iy - iy_ne)
            w_se = (ix - ix_nw) * (iy - iy_nw)

            return _sum_tensors( get_summand(ix, iy, w) for (ix, iy, w) in ((ix_nw, iy_nw, w_nw), (ix_ne, iy_ne, w_ne), (ix_sw, iy_sw, w_sw),(ix_se, iy_se, w_se),))

class ParticleWeigherAffineMapEncoderUnscaled(ParticleWeigherDualEncoderUnscaled):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherAffineMapEncoderUnscaled, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

    def create_input_processor(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        
        # Create an input processor for this model
        self.input_processor = ParticleWeigherAffineTransformerInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)

class ParticleWeigherAffineMapEncoder(ParticleWeigherDualEncoder):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherAffineMapEncoder, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

    def create_input_processor(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        
        # Create an input processor for this model
        self.input_processor = ParticleWeigherAffineTransformerInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)

class ParticleWeigherAffineMapEncoderUnboundPositive(ParticleWeigherDualEncoderUnboundPositive):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherAffineMapEncoderUnboundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

    def create_input_processor(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        
        # Create an input processor for this model
        self.input_processor = ParticleWeigherAffineTransformerInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)

class ParticleWeigherAffineMapEncoderBoundPositive(ParticleWeigherDualEncoderBoundPositive):
    def __init__(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        super(ParticleWeigherAffineMapEncoderBoundPositive, self).__init__(model_parameters, observation_encoder, particle_encoder, particle_transformer)

    def create_input_processor(self, model_parameters, observation_encoder, particle_encoder, particle_transformer):
        
        # Create an input processor for this model
        self.input_processor = ParticleWeigherAffineTransformerInputProcessor(model_parameters, observation_encoder, particle_encoder, particle_transformer)






def create_particle_weight_model(model_name, model_parameters, observation_encoder, particle_encoder, particle_transformer):

    model_type = model_parameters[model_name]["type"]

    if(model_type == "ParticleWeigher"):
        parameters = model_parameters[model_name]
        return ParticleWeigher(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherUnscaled"):        
        parameters = model_parameters[model_name]
        return ParticleWeigherUnscaled(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleUnboundPositive"):
        parameters = model_parameters[model_name]
        return ParticleUnboundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleBoundPositive"):
        parameters = model_parameters[model_name]
        return ParticleBoundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherDualEncoder"):
        parameters = model_parameters[model_name]
        return ParticleWeigherDualEncoder(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherDualEncoderUnscaled"):        
        parameters = model_parameters[model_name]
        return ParticleWeigherDualEncoderUnscaled(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherDualEncoderUnboundPositive"):
        parameters = model_parameters[model_name]
        return ParticleWeigherDualEncoderUnboundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherDualEncoderBoundPositive"):
        parameters = model_parameters[model_name]
        return ParticleWeigherDualEncoderBoundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherAffineMapEncoderUnscaled"):
        parameters = model_parameters[model_name]
        return ParticleWeigherAffineMapEncoderUnscaled(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherAffineMapEncoder"):
        parameters = model_parameters[model_name]
        return ParticleWeigherAffineMapEncoder(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherAffineMapEncoderUnboundPositive"):
        parameters = model_parameters[model_name]
        return ParticleWeigherAffineMapEncoderUnboundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherAffineMapEncoderBoundPositive"):
        parameters = model_parameters[model_name]
        return ParticleWeigherAffineMapEncoderBoundPositive(parameters, observation_encoder, particle_encoder, particle_transformer)

    elif(model_type == "ParticleWeigherHouse3D"):
        parameters = model_parameters[model_name]
        return ParticleWeigherHouse3D(parameters, observation_encoder, particle_encoder, particle_transformer)

    else:
        print("Unknown weight_model type \"{}\"".format(model_type))
        exit()



