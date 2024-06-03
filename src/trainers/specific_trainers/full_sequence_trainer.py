
#Standard Imports 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from functools import partial

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *

# Project Imports
from trainers.trainer import *
from problems.problem_base import *
from trainers.trainer_utils import LossShaper


class FullSequenceTrainer(Trainer):
    def __init__(self, model, problem, params, save_dir, device):
        super().__init__(params, model, problem, save_dir, device)

        # We need to know how many particles to train with
        assert("number_of_particles" in self.training_params)
        self.number_of_particles = self.training_params["number_of_particles"]

        # If we should skip the first step in the sequence when computing the loss 
        if("skip_first_step_for_loss" in self.training_params):
            self.skip_first_step_for_loss = self.training_params["skip_first_step_for_loss"]
        else:
            self.skip_first_step_for_loss = False

        # The max sequence length to use.  If not set then we use the one returned by the dataloader
        if("max_sequence_length" in self.training_params):
            self.max_sequence_length = self.training_params["max_sequence_length"]
        else:
            self.max_sequence_length = None

        if("truncated_bptt_modulo" in self.training_params):
            self.truncated_bptt_modulo = self.training_params["truncated_bptt_modulo"]
        else:
            self.truncated_bptt_modulo = None

        if("loss_shaping_params" in self.training_params):
            self.loss_shaper = LossShaper(self.training_params["loss_shaping_params"])
        else:
            self.loss_shaper = None

        self.bandwidths_for_plotting = dict()

        self.particle_variances = None
        self.particle_iqrs = None
        self.render_counter = 0

    def get_training_type(self):
        return "full"

    def do_freeze_rnn_batchnorm_layers(self):
        return True

    def do_forward_pass(self, data, dataset, epoch):

        data = self.convert_to_model_dtype(data)

        # Unpack the data and move to the device
        observations = data["observations"].to(self.device, non_blocking=True)
        states = data["states"].to(self.device, non_blocking=True)

        if("ground_truth_mask" in data):
            ground_truth_mask = data["ground_truth_mask"].to(self.device, non_blocking=True)
        else:
            ground_truth_mask = None

        # Sometimes we dont have actions
        if("actions" in data):
            actions = data["actions"]
            if(actions is not None):
                actions = actions.to(self.device, non_blocking=True)
        else:
            actions = None

        # Sometimes we dont have reference patches
        if("reference_patch" in data):
            reference_patch = data["reference_patch"]
            if(reference_patch is not None):
                reference_patch = reference_patch.to(self.device, non_blocking=True)
        else:
            reference_patch = None

        # Sometimes we dont have reference patches
        if("world_map" in data):
            world_map = data["world_map"]
            if(world_map is not None):
                world_map = world_map.to(self.device, non_blocking=True)
                # world_map = [w.to(self.device, non_blocking=True) for w in world_map]
        else:
            world_map = None


        # If we have a max sequence length then we need to chop everything down
        if(self.max_sequence_length is not None):
            ml = min(self.max_sequence_length, observations.shape[1])


            def chop(x):
                if(x is None):
                    return x
                else:
                    return x[:, :ml, ...]

            observations = chop(observations)
            states = chop(states)
            ground_truth_mask = chop(ground_truth_mask)
            actions = chop(actions)
            reference_patch = chop(reference_patch)
            # No need to chop the world map


        # Extract some statistics
        batch_size = observations.shape[0]
        subsequence_length = observations.shape[1]

        # transform the observation
        transformed_observation = self.problem.observation_transformer.forward_tranform(observations)


        # Run through the sub-sequence and compute outputs/losses
        all_losses = []
        for seq_idx in range(subsequence_length):

            if(seq_idx == 0):

                # Create the initial state for the particle filter
                output_dict = self.main_model.create_initial_dpf_state(states[:,0,:],transformed_observation, self.number_of_particles)

            else:

                # Get the observation for this step
                observation = transformed_observation[:,seq_idx,:]

                # Get the observation for the next step (if there is a next step)
                if((seq_idx+1) >= subsequence_length):
                    next_observation = None
                else:
                    next_observation = transformed_observation[:,seq_idx+1,:]

                # Make a copy of everything in the output dict
                output_dict_copy = dict()
                for key in output_dict.keys():
                    if(torch.is_tensor(output_dict[key])):
                        output_dict_copy[key] = output_dict[key].clone()
                    else:
                        output_dict_copy[key] = output_dict[key]


                # Create the next input dict from the last output dict
                input_dict = output_dict_copy
                input_dict["observation"] = observation
                input_dict["next_observation"] = next_observation
                input_dict["reference_patch"] = reference_patch
                input_dict["world_map"] = world_map

                if(actions is not None):
                    input_dict["actions"] = actions[:,seq_idx-1, :]
                else:
                    input_dict["actions"] = None

                # We have no timestep information
                input_dict["timestep_number"] = seq_idx

                # Do truncated BPTT 
                if(self.truncated_bptt_modulo is not None):
                    if((seq_idx != 1) and (((seq_idx-1) % self.truncated_bptt_modulo) == 0)):

                        for key in input_dict.keys():
                            if(torch.is_tensor(input_dict[key]) == False):
                                continue
                            input_dict[key] = input_dict[key].detach()

                # if(("particles" in input_dict) and input_dict["particles"].requires_grad):
                    # input_dict["particles"].register_hook(lambda grad: print("particle", torch.norm(grad.detach(), 2.0)))

                # if(("particle_weights" in input_dict) and input_dict["particle_weights"].requires_grad):
                    # input_dict["particle_weights"].register_hook(lambda grad: print("weights", torch.norm(grad.detach(), 2.0)))


                input_dict = self.convert_to_model_dtype(input_dict)

                # Run the model on this step
                output_dict = self.main_model(input_dict)

            # By default compute the loss
            compute_loss = True

            # If we are in the first step and we inited with the true state then the loss is a wash and should be ignored
            if(seq_idx == 0):
                if(self.main_model.initilize_with_true_state or self.skip_first_step_for_loss):
                    compute_loss = False

            # Compute the loss if the flag is true
            if(compute_loss):

                loss = self.loss_function.compute_loss(output_dict, states[:,seq_idx,:])

                # Shape the loss if needed
                if(self.loss_shaper is not None):
                    loss = self.loss_shaper.shape_loss(loss, seq_idx)


                # Mask if needed
                if(ground_truth_mask is not None):
                    
                    if(torch.sum(ground_truth_mask[:, seq_idx]).item() == 0):
                        # Nothing to do here since all the mask was false
                        continue                    

                    elif(torch.sum(ground_truth_mask[:, seq_idx]).item() == ground_truth_mask.shape[0]):
                        # Whole mask is the same and true
                        all_losses.append(loss)        

                    else:
                        # Mask is non zero but different per batch item so need to do this.  
                        mask = ground_truth_mask[:, seq_idx].float()
                        loss = loss * mask.unsqueeze(-1)
                        all_losses.append(loss)

                else:
                    all_losses.append(loss)



        # Aggregate the losses
        all_losses = torch.stack(all_losses)
        average_loss = torch.mean(all_losses)


        # Step the loss shaper
        if(self.loss_shaper is not None):
            self.loss_shaper.step()

        return average_loss, batch_size




    def convert_to_model_dtype(self, data):

        # for k in data.keys():
        #     if(torch.is_tensor(data[k])):
        #         data[k] = data[k].to(self.main_model.dtype)

        return data