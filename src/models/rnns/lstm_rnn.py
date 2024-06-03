# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import functools

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# For drawing the compute graphs
import torchviz

# Project Imports
from models.particle_transformer import *

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *
from kernel_density_estimation.epanechnikov import *
from kernel_density_estimation.circular_epanechnikov import *

# Other Models
from models.sequential_models import *
from models.internal_models.observation_encoder_models import *
from models.internal_models.particle_encoder_models import *
from models.internal_models.initializer_models import *
from models.internal_models.bandwidth_models import *



class LSTMInternal(nn.Module):
    def __init__(self, model_name, model_architecture_params):
        super(LSTMInternal, self).__init__()

        # Extract the specific parmas for this model
        model_params = model_architecture_params[model_name]

        # Extract the stuff we need to create the model
        input_dim = model_params["input_dim"]
        output_dim = model_params["output_dim"]
        self.number_of_lstm_layers = model_params["number_of_lstm_layers"]
        self.internal_latent_space_dim = model_params["internal_latent_space_dim"]
        number_of_fc_layers = model_params["number_of_fc_layers"]
        non_linear_type = model_params["non_linear_type"]

        # Check some things
        assert(input_dim > 0)
        assert(output_dim > 0)
        assert(self.number_of_lstm_layers > 0)
        assert(self.internal_latent_space_dim > 0)
        assert(number_of_fc_layers > 0)

        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU   
        else:
            assert(False)

        # Create the LSTM layers
        self.lstm_layers = nn.LSTM(input_size=input_dim, hidden_size=self.internal_latent_space_dim, num_layers=self.number_of_lstm_layers, batch_first=True)

        # All the FC layer stuff
        fc_layers = []

        # Create the intermediate layers FC layers
        if(number_of_fc_layers > 1):
            for i in range(number_of_fc_layers-1):
                fc_layers.append(nn.Linear(in_features=self.internal_latent_space_dim, out_features=self.internal_latent_space_dim))
                fc_layers.append(non_linear_object())

        # Add the final FC layer
        # fc_layers.append(non_linear_object())
        fc_layers.append(nn.Linear(in_features=self.internal_latent_space_dim, out_features=output_dim))

        # Covert the FC layers from list of a sequntial
        self.fc_layers = nn.Sequential(*fc_layers)


    def forward(self, hidden_state, final_lstm_input):

        # Run the LSTM 
        out, new_hidden_state = self.lstm_layers(final_lstm_input.unsqueeze(1), hidden_state)

        # Remove the sequence length part of the shape since we operate on timestep at a time
        out = out.squeeze(1)

        # Run the fc layers
        out = self.fc_layers(out)

        return out, new_hidden_state


    def create_initial_input_hidden_state(self, batch_size, device):

        # Create the 2 parts of the hidden state
        h0 = torch.zeros((self.number_of_lstm_layers, batch_size,  self.internal_latent_space_dim), device=device)
        c0 = torch.zeros((self.number_of_lstm_layers, batch_size,  self.internal_latent_space_dim), device=device)

        # Apparently they need to be in a tuple
        hidden_state = (h0, c0)

        return hidden_state


class LSTMRnn(SequentialModels):

    def __init__(self, model_params, model_architecture_params):
        super(LSTMRnn, self).__init__(model_params, model_architecture_params)

        # Extract the main model parameters
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]

        # Extract some parameters
        self.encode_particles = main_model_arch_params["encode_particles"]

        # Get the parameters for the kde
        assert("kde" in main_model_arch_params)
        self.kde_params = main_model_arch_params["kde"]

        if("use_local_maps" in main_model_arch_params):
            self.use_local_maps = main_model_arch_params["use_local_maps"]

            # the global map will be down-scaled by some factor (must be a float)
            self.window_scaler = float(8.0)
        
            # The size of the local map (must be an int)
            self.local_map_size = int(28)

        else:
            self.use_local_maps  = False

        # Create the internal model parameters
        # Note we dont need the particle dim in this case
        self.create_internal_models(model_params, model_architecture_params, None)

        # Get if we need a particle transformer for this model
        # If not then we will use the identity transformer (aka does not transform)
        if("particle_transformer" in main_model_arch_params):
            particle_transformer_params =  main_model_arch_params["particle_transformer"]
            self.particle_transformer = create_particle_transformer(particle_transformer_params)
        else:
            self.particle_transformer = IdentityTransformer()

        # Check what kind of observation encoder we want to use
        self.observation_is_patches = main_model_arch_params["observation_is_patches"]

        if(self.observation_is_patches):
            # Make sure the parameters we need are present
            assert("bounding_box_center_x_state_dim_pos" in main_model_arch_params)
            assert("bounding_box_center_y_state_dim_pos" in main_model_arch_params)
            assert("bounding_box_width_state_dim_pos" in main_model_arch_params)
            assert("bounding_box_height_state_dim_pos" in main_model_arch_params)
            assert("state_scale" in main_model_arch_params)

            # Extract the parameters
            self.bounding_box_center_x_state_dim_pos = main_model_arch_params["bounding_box_center_x_state_dim_pos"]
            self.bounding_box_center_y_state_dim_pos = main_model_arch_params["bounding_box_center_y_state_dim_pos"]
            self.bounding_box_width_state_dim_pos = main_model_arch_params["bounding_box_width_state_dim_pos"]
            self.bounding_box_height_state_dim_pos = main_model_arch_params["bounding_box_height_state_dim_pos"]
            self.state_scale = main_model_arch_params["state_scale"]

            # Get the image size we should reshape into.  The reshaped patch will be a square!
            self.patch_image_size = self.observation_encoder.input_image_size

            # Create the grids we will be using to crop and resize the patch into
            ls = torch.linspace(0, 1, self.patch_image_size)
            self.x_grid, self.y_grid = torch.meshgrid(ls, ls, indexing="ij")

            # Create the grids we will be using to resize the reference patch into
            ls = torch.linspace(-1, 1, self.patch_image_size)
            self.x_grid_ref_patches, self.y_grid_ref_patches = torch.meshgrid(ls, ls, indexing="ij")

        else:

            # Extract the number of obs we need
            assert("number_of_observation_inputs" in main_model_arch_params)
            self.number_of_observation_inputs = main_model_arch_params["number_of_observation_inputs"]
            assert((self.number_of_observation_inputs == 1) or (self.number_of_observation_inputs == 2))

        if("residual_scale_factor" in main_model_arch_params):
            self.residual_scale_factor = main_model_arch_params["residual_scale_factor"]

            # If its a list then it needs to become a tensor
            if(isinstance(self.residual_scale_factor, list)):
                self.residual_scale_factor = torch.FloatTensor(self.residual_scale_factor)
                self.residual_scale_factor = self.residual_scale_factor.unsqueeze(0)
        else:
            self.residual_scale_factor = 1.0



    def outputs_kde(self):
        return True

    def outputs_particles_and_weights(self):
        return True

    def outputs_single_solution(self):
        return False

    def create_internal_models(self, model_params, model_architecture_params, particle_dims):

        # Create the internal LSTM portion of this model
        self.lstm_internal = LSTMInternal(model_params["main_model"], model_architecture_params)

        # All the different networks we can have.  If we dont have a specific network then it is set to None
        # (aka the None value is never overridden)
        self.observation_encoder = None
        self.particle_encoder_for_particles_model = None
        self.particle_encoder_for_weights_model = None
        self.action_encoder_model = None
        self.initializer_model = None
        self.weighted_bandwidth_predictor = None

        # Get the sub models 
        sub_models = model_params["sub_models"]

        # Load the initializer model
        if("initializer_model" in sub_models):
            model_name = sub_models["initializer_model"]
            self.initializer_model = create_initializer_model(model_name, model_architecture_params)

        # Load the observation encoder model
        if("observation_model" in sub_models):
            model_name = sub_models["observation_model"]
            self.observation_encoder = create_observation_model(model_name, model_architecture_params)
        else:
            print("Need to specify a observation model")
            exit()

        # Get the particle encoder if we need one
        if(self.encode_particles):
            if("particle_encoder_for_particles_model" in sub_models):
                model_name = sub_models["particle_encoder_for_particles_model"]
                self.particle_encoder_for_particles_model = create_particle_encoder_model(model_name, model_architecture_params)
            else:
                print("Need to specify a particle encoder model")
                exit()

            # If we are encoding then we may need the particle encoder model
            if("action_encoder_model" in sub_models):
                model_name = sub_models["action_encoder_model"]
                self.action_encoder_model = create_particle_encoder_model(model_name, model_architecture_params)
            else:
                self.action_encoder_model = None    

            if(self.use_local_maps):
                if("particle_encoder_for_weights_model" in sub_models):
                    model_name = sub_models["particle_encoder_for_weights_model"]
                    self.particle_encoder_for_weights_model = create_particle_encoder_model(model_name, model_architecture_params)
                else:
                    print("Need to specify a particle encoder model for the weights")
                    exit()
            else:
                self.particle_encoder_for_weights_model = None

        else:
            self.particle_encoder_for_particles_model = None
            self.particle_encoder_for_weights_model = None
            self.action_encoder_model = None

            if("particle_encoder_for_particles_model" in sub_models):
                print("WARNING WARNING WARNING WARNING WARNING WARNING ")
                print("Particle encoder specified BUT flag to encode particles is not set.  Will not be using particle encoder")

            if("particle_encoder_for_weights_model" in sub_models):
                print("WARNING WARNING WARNING WARNING WARNING WARNING ")
                print("Particle encoder specified BUT flag to encode particles is not set.  Will not be using particle encoder")

            if("action_encoder_model" in sub_models):
                print("WARNING WARNING WARNING WARNING WARNING WARNING ")
                print("Action encoder specified BUT flag to encode particles is not set.  Will not be using action encoder")


        # Load the bandwidth predictor model
        if("bandwidth_model" in sub_models):
            model_name = sub_models["bandwidth_model"]
            self.weighted_bandwidth_predictor = create_bandwidth_model(model_name, model_architecture_params)
        else:
            print("Need to specify a bandwidth model")
            exit()


    def create_and_add_optimizers(self, training_params, trainer, training_type):

        if(self.initializer_model is not None):
            if("initializer_model_learning_rate" in training_params):
                lr = training_params["initializer_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.initializer_model], ["initializer_model"], lr)

        if(self.observation_encoder is not None):
            if("observation_model_learning_rate" in training_params):
                lr = training_params["observation_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.observation_encoder], ["observation_encoder"], lr)

        if(self.particle_encoder_for_particles_model is not None):
            if("particle_encoder_for_particles_learning_rate" in training_params):
                lr = training_params["particle_encoder_for_particles_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.particle_encoder_for_particles_model], ["particle_encoder_for_particles_model"], lr)

        if((self.particle_encoder_for_weights_model is not None) and self.particle_encoder_for_weights_model.is_learned_model()):
            if("particle_encoder_for_weights_learning_rate" in training_params):
                lr = training_params["particle_encoder_for_weights_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.particle_encoder_for_weights_model], ["particle_encoder_for_weights_model"], lr)

        if(self.action_encoder_model is not None):
            if("action_encoder_model_learning_rate" in training_params):
                lr = training_params["action_encoder_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.action_encoder_model], ["action_encoder_model"], lr)

        if("lstm_internal_learning_rate" in training_params):
            lr = training_params["lstm_internal_learning_rate"]
            trainer.add_optimizer_and_lr_scheduler([self.lstm_internal], ["lstm_internal"], lr)


        if((self.weighted_bandwidth_predictor is not None) and self.weighted_bandwidth_predictor.is_learned_model()):
            if("bandwidth_model_learning_rate" in training_params):
                lr = training_params["bandwidth_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.weighted_bandwidth_predictor], ["weighted_bandwidth_predictor"], lr)

    def add_models(self, trainer, training_type):
        if(training_type == "full"):
            trainer.add_model(self, "full_rnn_model")
            trainer.add_model_for_training("full_rnn_model")

            # add all the individual models so that we can load them individually as needed

            trainer.add_model(self.lstm_internal, "lstm_internal")

            if(self.initializer_model is not None):
                trainer.add_model(self.initializer_model, "initializer_model")

            if(self.observation_encoder is not None):
                trainer.add_model(self.observation_encoder, "observation_encoder")

            if(self.particle_encoder_for_particles_model is not None):
                trainer.add_model(self.particle_encoder_for_particles_model, "particle_encoder_for_particles_model")

            if(self.particle_encoder_for_weights_model is not None):
                trainer.add_model(self.particle_encoder_for_weights_model, "particle_encoder_for_weights_model")

            if(self.action_encoder_model is not None):
                trainer.add_model(self.action_encoder_model, "action_encoder_model")

            
            if((self.weighted_bandwidth_predictor is not None) and self.weighted_bandwidth_predictor.is_learned_model()):
                trainer.add_model(self.weighted_bandwidth_predictor, "weighted_bandwidth_predictor")

        else:
            print("Unknown Training Type: {}".format(training_type))
            assert(False)

    def load_pretrained(self, pre_trained_models, device):

        if("dpf_model" in pre_trained_models):
            state_dict = torch.load(pre_trained_models["dpf_model"], map_location=device)
            self.load_state_dict(state_dict)

            print("Loading {}".format(pre_trained_models["dpf_model"]))

        if(("initializer_model" in pre_trained_models) and (self.initializer_model is not None)):
            self.initializer_model.load_state_dict(torch.load(pre_trained_models["initializer_model"], map_location=device))
        else:
            print("Not loading for \"initializer_model\"")

        if("observation_model" in pre_trained_models):
            self.observation_encoder.load_state_dict(torch.load(pre_trained_models["observation_model"], map_location=device))
        else:
            print("Not loading for \"observation_model\"")

        if("lstm_internal" in pre_trained_models):
            self.lstm_internal.load_state_dict(torch.load(pre_trained_models["lstm_internal"], map_location=device))
        else:
            print("Not loading for \"lstm_internal\"")

        if(("particle_encoder_for_particles_model" in pre_trained_models) and (self.particle_encoder_for_particles_model is not None)):
            self.particle_encoder_for_particles_model.load_state_dict(torch.load(pre_trained_models["particle_encoder_for_particles_model"], map_location=device))
        else:
            print("Not loading for \"particle_encoder_for_particles_model\"")

        if(("particle_encoder_for_weights_model" in pre_trained_models) and (self.particle_encoder_for_weights_model is not None)):
            self.particle_encoder_for_weights_model.load_state_dict(torch.load(pre_trained_models["particle_encoder_for_weights_model"], map_location=device))
        else:
            print("Not loading for \"particle_encoder_for_weights_model\"")

        if(("action_encoder_model" in pre_trained_models) and (self.action_encoder_model is not None)):                
            self.action_encoder_model.load_state_dict(torch.load(pre_trained_models["action_encoder_model"], map_location=device))
        else:
            print("Not loading for \"action_encoder_model\"")

        if(("bandwidth_model" in pre_trained_models) and (self.weighted_bandwidth_predictor is not None)):
            data = torch.load(pre_trained_models["bandwidth_model"], map_location=device)

            if(isinstance(data, dict)):
                self.weighted_bandwidth_predictor.load_state_dict(data)
            else:
                self.weighted_bandwidth_predictor = data
        else:
            print("Not loading for \"bandwidth_model\"")

    def freeze_rnn_batchnorm_layers(self):

        # Freeze all models that have batchnorm that are in the rnn path

        if(self.initializer_model is not None):
            self.initializer_model.freeze_batchnorms()

        if(self.particle_encoder_for_particles_model is not None):
            self.particle_encoder_for_particles_model.freeze_batchnorms()

        if(self.particle_encoder_for_weights_model is not None):
            self.particle_encoder_for_weights_model.freeze_batchnorms()

        if(self.action_encoder_model is not None):
            self.action_encoder_model.freeze_batchnorms()

    def create_initial_dpf_state(self, true_state, observations, particle_dims):

        # Need the batch size
        batch_size = true_state.shape[0]

        if(self.initilize_with_true_state):

            # Set the predicted state to be the true state and then create the hidden state with zeros
            predicted_state = true_state.detach().clone().to(observations.device)

            # Add noise to the initial particles
            assert(len(self.initial_position_std) == len(self.kde_params["dims"]))
            for d in range(len(self.initial_position_std)):

                # Get the stats and the dist type
                std = self.initial_position_std[d]
                dim_params = self.kde_params["dims"][d]

                # create the correct dist for this dim
                distribution_type = dim_params["distribution_type"]
                if(distribution_type == "Normal"):
                    dist = D.Normal(loc=torch.zeros_like(predicted_state[..., d]),  scale=std)
                elif(distribution_type == "Epanechnikov"):
                    dist = Epanechnikov(loc=torch.zeros_like(predicted_state[..., d]),  bandwidth=std)
                elif(distribution_type == "Von_Mises"):
                    kappa = 1.0 / std
                    dist = VonMisesFullDist(loc=torch.zeros_like(predicted_state[..., d]), concentration=kappa)

                # Generate and add some noise
                noise = dist.sample()
                predicted_state[..., d] = predicted_state[..., d] + noise

        else:           

            assert(False)

            # # Extract the first observation 
            # obs = observations[:, 0, :]
            # encoded_obs = self.observation_encoder(obs)

            # # Create the initial state guess with just 1 particle
            # predicted_state = self.initializer_model(encoded_obs, 1).squeeze(1)

            # # Norm the predicted_state.  Most of the time this wont do anything but sometimes we want to 
            # # norm some dims
            # predicted_state = self.particle_transformer.apply_norm(predicted_state)

            # # Convert from internal dim to output dim
            # predicted_state = self.particle_transformer.forward_tranform(predicted_state)

        # Set the hidden state to be zeros
        hidden_state = self.lstm_internal.create_initial_input_hidden_state(batch_size, predicted_state.device)

        # Pack everything into an output dict
        output_dict = dict()
        output_dict["hidden_state"] = hidden_state
        output_dict["predicted_state"] = predicted_state


        # Create particles
        particles = predicted_state.unsqueeze(1)
        particle_weights = torch.ones_like(particles[:, :, 0])

        if(self.initilize_with_true_state):
            bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1])).to(observations.device)
            bandwidths[...] = 0.001

        else:
            bandwidths = self.weighted_bandwidth_predictor(particles, particle_weights)

        output_dict["particles_downscaled"] = particles
        output_dict["particles"] = self.particle_transformer.upscale(particles)
        output_dict["particle_weights"] = particle_weights
        output_dict["bandwidths_downscaled"] = bandwidths
        output_dict["bandwidths"] = self.particle_transformer.upscale(bandwidths)

        return output_dict



    def create_current_state_action_inputs(self, predicted_state, actions):

        if(self.encode_particles):

            # encode the state
            encoded_predicted_state = self.particle_encoder_for_particles_model(predicted_state.unsqueeze(1)).squeeze(1)

            # Encode the actions if we need to
            if(self.action_encoder_model is not None):

                # Encode the actions
                encoded_actions = self.action_encoder_model(actions.unsqueeze(1)).squeeze(1)

                # If we encoded the action then concat it and return a single large vector
                return torch.cat([encoded_predicted_state, encoded_actions], dim=-1)
                
            else:
                #  No actions so just return the encoded state
                return encoded_predicted_state

        else:
            # No encoding so if we dont have an action return the state otherwise concat the action and state into 1 vector
            if(actions is None):
                return predicted_state
            else:
                return torch.cat([predicted_state, actions], dim=-1)

    def create_encoded_observation_input(self, observations, next_observation, reference_patch, predicted_state):

        # if we have no observation then there is nothing to weight so no need to prepare any of the inputs
        if(observations is None):
            return None

        if(self.observation_is_patches):

            # We need at least 4 dims to specify the bounding box
            assert(predicted_state.shape[-1] >= 4)

            # Extract the patches
            extracted_patches = self.extract_and_resize_patches(predicted_state, observations)

            # Encode the patches
            encoded_extracted_patches = self.observation_encoder(extracted_patches)        

            # Resize the reference patches to the correct size
            resized_reference_patches = self.resize_reference_patches(reference_patch)

            # Encode the reference patches
            encoded_reference_patches = self.observation_encoder(resized_reference_patches)

            # Encode the and add the observations
            final_weight_net_inputs = []
            final_weight_net_inputs.append(encoded_reference_patches)
            final_weight_net_inputs.append(encoded_extracted_patches)

            # The final input is all the encoding parts put into 1 big vector 
            return torch.cat(final_weight_net_inputs,dim=-1)

        else:

            # If we need the next observation and we dont have it then there is nothing to prepare
            if((self.number_of_observation_inputs == 2) and (next_observation is None)):
                return None

            encoded_observation = self.observation_encoder(observations)

            if(self.number_of_observation_inputs == 2):
                encoded_next_observation = self.observation_encoder(next_observation)
                return torch.cat([encoded_observation, encoded_next_observation], dim=-1)

            else:
                return encoded_observation                

    def extract_and_resize_patches(self, predicted_state, observations):

        # Rescale predicted_state to be between -1 and 1
        predicted_state = predicted_state / self.state_scale

        # Make sure the grids are on the correct device
        if((self.x_grid.device != predicted_state.device) or (self.y_grid.device != predicted_state.device)):
            self.x_grid = self.x_grid.to(predicted_state.device)
            self.y_grid = self.y_grid.to(predicted_state.device)

        # Tile the grids so we have 1 per batch
        tiled_x_grid = torch.tile(self.x_grid.unsqueeze(0), (predicted_state.shape[0], 1, 1))
        tiled_y_grid = torch.tile(self.y_grid.unsqueeze(0), (predicted_state.shape[0], 1, 1))

        # extract the widths and heights
        bb_widths = predicted_state[..., self.bounding_box_width_state_dim_pos]
        bb_heights = predicted_state[..., self.bounding_box_height_state_dim_pos]

        # Compute the sampling locations
        tiled_x_grid = tiled_x_grid * bb_widths.unsqueeze(-1).unsqueeze(-1)
        tiled_y_grid = tiled_y_grid * bb_heights.unsqueeze(-1).unsqueeze(-1)
        tiled_x_grid = tiled_x_grid + (predicted_state[..., self.bounding_box_center_x_state_dim_pos] - (bb_widths / 2)).unsqueeze(-1).unsqueeze(-1)
        tiled_y_grid = tiled_y_grid + (predicted_state[..., self.bounding_box_center_y_state_dim_pos] - (bb_heights / 2)).unsqueeze(-1).unsqueeze(-1)

        # Rescale to be be within [-1, 1]
        # tiled_x_grid = ((tiled_x_grid / observations.shape[-1]) * 2.0) - 1.0
        # tiled_y_grid = ((tiled_y_grid / observations.shape[-2]) * 2.0) - 1.0

        zeros_array = torch.zeros_like(tiled_x_grid)

        grid = torch.stack([tiled_x_grid, tiled_y_grid, zeros_array], dim=-1)

        # tiled_observations = torch.tile(observations.unsqueeze(2), (1, 1, predicted_state.shape[1], 1, 1))
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

    def forward(self, input_dict):

        # unpack the inputs
        predicted_state = input_dict["predicted_state"]
        hidden_state = input_dict["hidden_state"]
        observation = input_dict["observation"]
        actions = input_dict["actions"]
        reference_patch = input_dict["reference_patch"]

        # we might have the next observation, if we do then unpack it
        if("next_observation" in input_dict):
            next_observation = input_dict["next_observation"]
        else:
            next_observation = None

        if("world_map" in input_dict):
            world_map = input_dict["world_map"]
        else:
            world_map = None



        if(self.use_local_maps):
            final_lstm_input =  self.process_local_maps(world_map, observation, predicted_state.unsqueeze(1))

            # Need to transform the particles from output space to internal space
            predicted_state = self.particle_transformer.backward_tranform(predicted_state)

            # Create the state/action input 
            state_action_input = self.create_current_state_action_inputs(predicted_state, actions)

            final_lstm_input = torch.cat([state_action_input, final_lstm_input], dim=-1)
        else:
            # Extract information about the input
            batch_size = observation.shape[0]

            # Need to transform the particles from output space to internal space
            predicted_state = self.particle_transformer.backward_tranform(predicted_state)

            # Create the state/action input 
            state_action_input = self.create_current_state_action_inputs(predicted_state, actions)

            # # Create the observation input 
            observation_input = self.create_encoded_observation_input(observation, next_observation, reference_patch, predicted_state)

            # Concat 
            final_lstm_input = torch.cat([state_action_input, observation_input], dim=-1)

        # Run the LSTM
        predicted_state_residual, new_hidden_state = self.lstm_internal(hidden_state, final_lstm_input)

        # Move it to the correct device
        if(isinstance(self.residual_scale_factor, torch.FloatTensor) and (self.residual_scale_factor.device != predicted_state_residual.device)):
            self.residual_scale_factor = self.residual_scale_factor.to(predicted_state_residual.device)

        # Scale the residual
        predicted_state_residual = (predicted_state_residual * self.residual_scale_factor)

        # Apply the residual
        predicted_state = predicted_state + predicted_state_residual

        # Norm the predicted_state.  Most of the time this wont do anything but sometimes we want to 
        # norm some dims
        predicted_state = self.particle_transformer.apply_norm(predicted_state)


        # Convert from internal dim to output dim
        predicted_state = self.particle_transformer.forward_tranform(predicted_state)


        # Pack everything into an output dict
        output_dict = dict()
        output_dict["hidden_state"] = new_hidden_state
        output_dict["predicted_state"] = predicted_state


        # Create particles
        particles = predicted_state.unsqueeze(1)
        particle_weights = torch.ones_like(particles[:, :, 0])

        # Predict bandwidths
        bandwidths = self.weighted_bandwidth_predictor(particles, particle_weights)

        output_dict["particles_downscaled"] = particles
        output_dict["particles"] = self.particle_transformer.upscale(particles)
        output_dict["particle_weights"] = particle_weights
        output_dict["bandwidths_downscaled"] = bandwidths
        output_dict["bandwidths"] = self.particle_transformer.upscale(bandwidths)

        return output_dict



    def process_local_maps(self, world_map, observations, particles):

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
        encoded_local_maps = self.particle_encoder_for_weights_model(local_maps)

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

        return final_weight_net_inputs


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
        translate_x = (translate_x * (2.0 / map_width)) - 1.0
        translate_y = (translate_y * (2.0 / map_height)) - 1.0

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
