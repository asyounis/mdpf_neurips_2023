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


class LearnedProposalBase(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(LearnedProposalBase, self).__init__(model_parameters)

        # Extract some parameters that we will be using a lot
        assert("input_particle_dimension" in model_parameters)
        self.input_particle_dimension = model_parameters["input_particle_dimension"]

        # Extract some parameters that we will be using a lot
        assert("output_particle_dimension" in model_parameters)
        self.output_particle_dimension = model_parameters["output_particle_dimension"]

        # Extract some parameters that we will be using a lot
        assert("noise_dimension" in model_parameters)
        self.noise_dimension = model_parameters["noise_dimension"]

    def create_proposal_model(self, model_parameters, output_dim, input_dim):

        # Make sure we have the correct parameters are passed in
        assert("proposal_latent_space" in model_parameters)
        assert("proposal_number_of_layers" in model_parameters)
        assert("proposal_encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed for the encoder
        proposal_latent_space = model_parameters["proposal_latent_space"]
        proposal_number_of_layers = model_parameters["proposal_number_of_layers"]
        proposal_encoder_use_batch_norm = model_parameters["proposal_encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]

        if("use_layer_norm" in model_parameters):
            use_layer_norm = model_parameters["use_layer_norm"]
        else:
            use_layer_norm = "None"

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
        assert(proposal_number_of_layers >= 2)

        # Create the encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=(input_dim),out_features=proposal_latent_space)))
        
        if(proposal_encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(proposal_latent_space))

        if(use_layer_norm == "pre_activation"):
            layers.append(nn.LayerNorm(proposal_latent_space))

        layers.append(non_linear_object())
        if(proposal_encoder_use_batch_norm == "post_activation"):
            layers.append(nn.BatchNorm1d(proposal_latent_space))

        if(use_layer_norm == "post_activation"):
            layers.append(nn.LayerNorm(proposal_latent_space))


        # the middle layers are all the same fully connected layers
        for i in range(proposal_number_of_layers-2):
            layers.append(self.apply_parameter_norm(nn.Linear(in_features=proposal_latent_space,out_features=proposal_latent_space)))
            if(proposal_encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(proposal_latent_space))

            if(use_layer_norm == "pre_activation"):
                layers.append(nn.LayerNorm(proposal_latent_space))


            layers.append(non_linear_object())
            if(proposal_encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(proposal_latent_space))

            if(use_layer_norm == "post_activation"):
                layers.append(nn.LayerNorm(proposal_latent_space))

        
        # Final layer is the output space and so does not need a non-linearity
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=proposal_latent_space, out_features=output_dim)))
        # if(proposal_encoder_use_batch_norm):
            # layers.append(nn.BatchNorm1d(output_dim))

        # Generate the model
        self.proposal_model = nn.Sequential(*layers)


class LearnedProposalNoTimestamp(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalNoTimestamp, self).__init__(model_parameters)

        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights


class LearnedProposalNoTimestampResidual(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalNoTimestampResidual, self).__init__(model_parameters)

        if("residual_scale_factor" in model_parameters):
            self.residual_scale_factor = model_parameters["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)

        else:
            self.residual_scale_factor = 1.0


        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        # out = torch.tanh(out)
        out = (torch.sigmoid(out) * 2.0) - 1.0


        # Move it to the correct device
        if(self.residual_scale_factor.device != out.device):
            self.residual_scale_factor = self.residual_scale_factor.to(out.device)

        # We want to learn the residual!
        out = (out * self.residual_scale_factor)
        out = particles + out

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights


class SyntheticTrackingSemiTrue(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(SyntheticTrackingSemiTrue, self).__init__(model_parameters)

        if("residual_scale_factor" in model_parameters):
            self.residual_scale_factor = model_parameters["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)

        else:
            self.residual_scale_factor = 1.0


        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        out = torch.tanh(out)

        # Move it to the correct device
        if(self.residual_scale_factor.device != out.device):
            self.residual_scale_factor = self.residual_scale_factor.to(out.device)

        # We want to learn the residual!
        out = (out*self.residual_scale_factor)


        particles[...,:2] = particles[...,:2] + particles[...,2:]

        out = particles + out

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights



class LearnedProposalBoundingBox(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalBoundingBox, self).__init__(model_parameters)

        if("residual_scale_factor" in model_parameters):
            self.residual_scale_factor = model_parameters["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)

        else:
            self.residual_scale_factor = 1.0


        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

        # Use to make sure the width and height of the bounding box is valid
        self.softplus = nn.Softplus()

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        out = torch.tanh(out)

        # Move it to the correct device
        if(self.residual_scale_factor.device != out.device):
            self.residual_scale_factor = self.residual_scale_factor.to(out.device)

        # We want to learn the residual!
        out = (out * self.residual_scale_factor)
        out = particles + out

        # Force the width and height of the predicted bounding box to be greater than 0
        # out[..., -2] = self.softplus(out[..., -2])
        # out[..., -1] = self.softplus(out[..., -1])
        # out = torch.cat([out[...,:-2], self.softplus(out[..., -2:])], dim=-1)
        out = torch.cat([out[...,:2], self.softplus(out[..., 2:4]), out[..., 4:]], dim=-1)

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights


class LearnedProposalBoundingBoxBound(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalBoundingBoxBound, self).__init__(model_parameters)

        if("residual_scale_factor" in model_parameters):
            self.residual_scale_factor = model_parameters["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)

        else:
            self.residual_scale_factor = 1.0


        self.output_mins = torch.FloatTensor(model_parameters["output_mins"])
        self.output_maxs = torch.FloatTensor(model_parameters["output_maxs"])

        # self.ranges = output_maxs - self.output_mins



        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

        # Use to make sure the width and height of the bounding box is valid
        self.softplus = nn.Softplus()


    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        # Move it to the correct device
        if(self.residual_scale_factor.device != out.device):
            self.residual_scale_factor = self.residual_scale_factor.to(out.device)

        # Scale the residual
        out = torch.tanh(out)
        out = (out * self.residual_scale_factor)

        # We want to learn the residual!
        out = particles + out

        # Force the width and height of the predicted bounding box to be greater than 0
        if(out.shape[-1] > 4):
            out = torch.cat([out[...,:2], self.softplus(out[..., 2:4]), out[..., 4:]], dim=-1)
        else:
            out = torch.cat([out[...,:2], self.softplus(out[..., 2:4]), ], dim=-1)

        # Move to the correct device if not already pn the correct device
        if((self.output_maxs.device != out.device) or (self.output_mins.device != out.device)):
            self.output_maxs = self.output_maxs.to(out.device)
            self.output_mins = self.output_mins.to(out.device)

        # Threshold the outputs
        out = torch.clamp(out, self.output_mins.unsqueeze(0).unsqueeze(0), self.output_maxs.unsqueeze(0).unsqueeze(0))

        # Scale the outputs
        # out = F.sigmoid(out)
        # out = out * self.ranges.unsqueeze(0).unsqueeze(0)
        # out = out + self.output_mins.unsqueeze(0).unsqueeze(0)

        # print(out.shape)
        # print(self.ranges.unsqueeze(0).unsqueeze(0))

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights


class LearnedProposalBound(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalBound, self).__init__(model_parameters)

        # Used to create random noise for the model to use
        self.random_noise_dist = D.Normal(0, 1)

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension, self.input_particle_dimension+self.noise_dimension)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # Create random noise and move it to the correct device
        # If we are passed in noise then use that
        if(noise is None):
            self.random_noise_dist = D.Normal(torch.as_tensor(0.0,device=encoded_particles.device), torch.as_tensor(1.0, device=encoded_particles.device))
            noise = self.random_noise_dist.sample((encoded_particles.shape[0], encoded_particles.shape[1], self.noise_dimension))
            # noise = noise.to(encoded_particles.device)

        # The final input is the encoding and noise
        final_input = torch.cat((encoded_particles, noise),dim=-1)
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        # out = torch.tanh(out)
        out = (torch.sigmoid(out) * 2.0) - 1.0

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights





class LearnedProposalNoTimestampResidualDistributionBase(LearnedProposalBase):
    def __init__(self, model_parameters):
        super(LearnedProposalNoTimestampResidualDistributionBase, self).__init__(model_parameters)

        if("residual_scale_factor" in model_parameters):
            self.residual_scale_factor = model_parameters["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0).unsqueeze(0)

        else:
            self.residual_scale_factor = 1.0

        # Create the proposal model
        self.create_proposal_model(model_parameters, self.output_particle_dimension*2, self.input_particle_dimension)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Make sure that the particles are the correct shape for this
        assert(encoded_particles.shape[-1] == self.input_particle_dimension)

        # The final input is the encoding that is flattened
        final_input = encoded_particles
        final_input = final_input.view(-1, final_input.shape[-1])

        # Do the forward pass
        out = self.proposal_model(final_input)

        # reshape back into the correct unflattened shape
        out = out.view(encoded_particles.shape[0], encoded_particles.shape[1], -1)

        # Move it to the correct device
        if(self.residual_scale_factor.device != out.device):
            self.residual_scale_factor = self.residual_scale_factor.to(out.device)

        # split them into means and stds
        d = out.shape[-1]
        means = out[..., :(d//2)]
        stds = out[..., (d//2):]

        # Scale the outputs
        means = (torch.sigmoid(means) * 2.0) - 1.0
        stds = torch.sigmoid(stds)

        # print(torch.mean(torch.mean(means, dim=0), dim=0))
        # print(torch.mean(torch.mean(stds, dim=0), dim=0))

        # Scale the outputs
        means = (means * self.residual_scale_factor)
        stds = (stds * self.residual_scale_factor)

        # Return the particle 
        return means, stds


class LearnedProposalNoTimestampResidualDistributionReparameterized(LearnedProposalNoTimestampResidualDistributionBase):
    def __init__(self, model_parameters):
        super(LearnedProposalNoTimestampResidualDistributionReparameterized, self).__init__(model_parameters)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Get the means and the weights
        means, stds = super().forward(particles, encoded_particles, encoded_observation, noise=None)


        # Draw samples from the means and variances
        dist = D.Normal(means, stds)
        samples = dist.rsample()

        # Add the samples to the particles to get the new particle location
        out = particles + samples

        # The un-normalized weights that we should use for this
        weights = torch.ones((out.shape[0], out.shape[1]), device=out.device)

        # Return the particle 
        return out, weights



class LearnedProposalNoTimestampResidualDistributionImportanceSampled(LearnedProposalNoTimestampResidualDistributionBase):
    def __init__(self, model_parameters):
        super(LearnedProposalNoTimestampResidualDistributionImportanceSampled, self).__init__(model_parameters)

    def forward(self, particles, encoded_particles, encoded_observation, noise=None):

        # Get the means and the weights
        means, stds = super().forward(particles, encoded_particles, encoded_observation, noise=None)

        # Draw samples from the means and variances
        dist = D.Normal(means, stds)
        samples = dist.sample()

        # Add the samples to the particles to get the new particle location
        out = particles + samples

        # The un-normalized weights that we should use for this
        weights = dist.log_prob(samples)
        weights = torch.sum(weights, dim=-1)

        # Do importance sampling weighting
        weights = torch.exp(weights - weights.detach())


        # Return the particle 
        return out.detach(), weights









def create_proposal_model(model_name, model_parameters):

    # 
    model_type = model_parameters[model_name]["type"]

    if(model_type == "TrueProposal"):
        print("Please Fix Me")
        exit()
        # self.proposal_model = TrueDynamics()

    elif(model_type == "TrueProposalLearnedNoise"):
        print("Please Fix Me")
        exit()
        # self.proposal_model = TrueDynamicsLearnedNoise()

    elif(model_type == "LearnedProposal"):
        # parameters = model_parameters[model_name]
        # return  LearnedProposal(parameters)

        print("Please Fix Me")
        exit()

    elif(model_type == "LearnedProposalMeanVariance"):
        # parameters = model_parameters[model_name]
        # return  LearnedProposalMeanVariance(parameters)

        print("Please Fix Me")
        exit()

    elif(model_type == "LearnedProposalNoTimestamp"):
        parameters = model_parameters[model_name]
        return  LearnedProposalNoTimestamp(parameters)

    elif(model_type == "LearnedProposalNoTimestampResidual"):
        parameters = model_parameters[model_name]
        return  LearnedProposalNoTimestampResidual(parameters)

    elif(model_type == "SyntheticTrackingSemiTrue"):
        parameters = model_parameters[model_name]
        return  SyntheticTrackingSemiTrue(parameters)

    elif(model_type == "LearnedProposalBoundingBox"):
        parameters = model_parameters[model_name]
        return  LearnedProposalBoundingBox(parameters)

    elif(model_type == "LearnedProposalBoundingBoxBound"):
        parameters = model_parameters[model_name]
        return  LearnedProposalBoundingBoxBound(parameters)

    elif(model_type == "LearnedProposalBound"):
        parameters = model_parameters[model_name]
        return  LearnedProposalBound(parameters)

    elif(model_type == "LearnedProposalNoTimestampResidualDistributionReparameterized"):
        parameters = model_parameters[model_name]
        return  LearnedProposalNoTimestampResidualDistributionReparameterized(parameters)

    elif(model_type == "LearnedProposalNoTimestampResidualDistributionImportanceSampled"):
        parameters = model_parameters[model_name]
        return  LearnedProposalNoTimestampResidualDistributionImportanceSampled(parameters)

    else:
        print("Unknown proposal_model type \"{}\"".format(model_type))
        exit()
