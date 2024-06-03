
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


class ProposalModelTrainer(Trainer):
    def __init__(self, model, problem, params, save_dir, device):
        super().__init__(params, model, problem, save_dir, device)

        # Extract the training parameters
        self.number_of_particles = self.training_params["number_of_particles"]

        if("number_of_particle_hidden_dims" in self.training_params):
            self.number_of_particle_hidden_dims = self.training_params["number_of_particle_hidden_dims"]
        else:
            self.number_of_particle_hidden_dims = 0


        # Make sure the datasets has at least 2 states
        assert(self.training_dataset.get_subsequence_length() >= 2)
        assert(self.validation_dataset.get_subsequence_length() >= 2)

        # Disable some outputs we dont care about to make things faster
        self.training_dataset.disable_output("world_map")
        self.validation_dataset.disable_output("world_map")

    def get_training_type(self):
        return "proposal"

    def do_forward_pass(self, data, dataset, epoch):

        # Unpack the data and move to the device
        states = data["states"].to(self.device)

        # We may or may not have actions
        if("actions" in data):
            actions = data["actions"].to(self.device)
        else:
            actions = None

        # Extract some statistics
        batch_size = states.shape[0]
        subsequence_length = states.shape[1]

        # Compute the loss as we go
        average_loss = 0

        # Run through the sub-sequence
        for seq_idx in range(subsequence_length-1):

            # Create the particle set
            particles = torch.tile(states[:,seq_idx,:].unsqueeze(1),[1,  self.number_of_particles, 1])

            # Add the number of dims we need to add, if we need to add any
            if(self.number_of_particle_hidden_dims != 0):

                # create the hidden state
                hidden_state = torch.zeros((particles.shape[0], particles.shape[1], self.number_of_particle_hidden_dims), device=particles.device)

                # append the hidden state to the particle state
                particles = torch.cat([particles, hidden_state], dim=-1)

            # Transform!!!!!
            particles = self.main_model.particle_transformer.backward_tranform(particles)


            # Encode the particles
            encoded_particles = self.main_model.particle_encoder_for_particles_model(particles)
            
            # If we have actions then encode them and add them to the encodings.  If not 
            # then the encodings are just the encoded particles
            if(actions is not None):
                # Encode the actions
                tiled_actions = torch.tile(actions[:,seq_idx,:].unsqueeze(1),[1,  self.number_of_particles, 1])
                encoded_actions = self.main_model.action_encoder_model(tiled_actions)

                # Concatenate the encoded particles and actions
                all_encodings = torch.cat([encoded_particles, encoded_actions], dim=-1)

            else:
                # If we haev no encoded actions then we just use the encoded particles
                all_encodings = encoded_particles


            if(torch.sum(torch.isnan(particles)) > 0):
                print("")
                print("No 1")
                exit()
            

            if(torch.sum(torch.isnan(all_encodings)) > 0):
                print("")
                print("No 1.1")
                exit()
            
            # Generate new particles
            new_particles, probs = self.main_model.proposal_model(particles, all_encodings, None)

            if(torch.sum(torch.isnan(new_particles)) > 0):
                print("")
                print("No 2")
                exit()
            


            # Decode the particle 
            new_particles = self.main_model.particle_transformer.forward_tranform(new_particles)

            if(torch.sum(torch.isnan(new_particles)) > 0):
                print("")
                print("No 3")
                exit()

            # Assume equally weighted particles for the KDE
            new_particle_weights = torch.ones(size=(batch_size, self.number_of_particles), device=self.device)
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
            loss = self.loss_function.compute_loss(output_dict, states[:,seq_idx+1,:])

            # Aggregate the loss via mean
            loss = torch.mean(loss)

            # Compute the average loss
            average_loss = average_loss + (loss /  (subsequence_length - 1))
            
        return average_loss, batch_size

















