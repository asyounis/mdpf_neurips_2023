#Standard Imports 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

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
from data_generators.data_generator_base import *
from problems.problem_base import *



class ParticleSetDataGenerator(DataGeneratorBase):
    def __init__(self, params, main_model, problem, save_dir, device):
        super().__init__(params, main_model, problem, save_dir, device)

        # We need to know how many particles to use
        assert("number_of_particles" in self.data_generator_params)
        self.number_of_particles = self.data_generator_params["number_of_particles"]

        # Parse the parameters
        self.epochs = self.data_generator_params["epochs"]
        self.batch_size = self.data_generator_params["batch_size"]
        
        if("num_cpu_cores_for_dataloader" in self.data_generator_params):
            self.num_cpu_cores_for_dataloader = self.data_generator_params["num_cpu_cores_for_dataloader"]
        else:
            self.num_cpu_cores_for_dataloader = 8

        # Get and create the dataset/loader. 
        self.dataset = self.problem.get_training_dataset()
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True)

        # All the output dict
        self.all_output_particles = []
        self.all_output_weights = []
        self.all_states = []

    def do_epoch(self, epoch):

        # Dont need the gradients for the evaluation epochs
        with torch.no_grad():

            self.main_model.eval()

            # Keep track of the average loss over this epoch
            average_loss = 0

            # Go through all the data once
            t = tqdm(iter(self.data_loader), leave=False, total=len(self.data_loader))
            for step, data in enumerate(t):

                # Do the forward pass over the data
                self.generate_batch_of_data(data, self.dataset, epoch)


    def generate_data(self):
        for epoch in tqdm(range(self.epochs)):
            self.do_epoch(epoch)


        # Convert into 1 big tensor
        self.all_output_particles = torch.vstack(self.all_output_particles)
        self.all_output_weights = torch.vstack(self.all_output_weights)
        self.all_states = torch.vstack(self.all_states)

        # Save
        torch.save(self.all_output_particles, "{}/particles.pt".format(self.save_dir))
        torch.save(self.all_output_weights, "{}/particle_weights.pt".format(self.save_dir))
        torch.save(self.all_states, "{}/states.pt".format(self.save_dir))

    def generate_batch_of_data(self, data, dataset, epoch):

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


        # Extract some statistics
        batch_size = observations.shape[0]
        subsequence_length = observations.shape[1]

        # transform the observation
        transformed_observation = self.problem.observation_transformer.forward_tranform(observations)


        # Run through the sub-sequence
        for seq_idx in range(subsequence_length):

            # If we should include this output dict in the dataset 
            do_include_output_dict = True

            if(seq_idx == 0):

                # Create the initial state for the particle filter
                output_dict = self.main_model.create_initial_dpf_state(states[:,0,:],transformed_observation, self.number_of_particles)

                # Only use if we are not initing with the true state
                if(self.main_model.initilize_with_true_state):
                    do_include_output_dict = False

            else:

                # Get the observation for this step
                observation = transformed_observation[:,seq_idx,:]

                # Get the observation for the next step (if there is a next step)
                if((seq_idx+1) >= subsequence_length):
                    next_observation = None
                else:
                    next_observation = transformed_observation[:,seq_idx+1,:]

                # Create the next input dict from the last output dict
                input_dict = output_dict
                input_dict["observation"] = observation
                input_dict["next_observation"] = next_observation
                input_dict["reference_patch"] = reference_patch

                if(actions is not None):
                    input_dict["actions"] = actions[:,seq_idx-1, :]
                else:
                    input_dict["actions"] = None

                # We have no timestep information
                input_dict["timestep_number"] = seq_idx

                # Run the model on this step
                output_dict = self.main_model(input_dict)

            # Save the relevant parts of the output dict
            if(do_include_output_dict):
                self.all_output_particles.append(output_dict["particles"].detach().cpu())
                self.all_output_weights.append(output_dict["particle_weights"].detach().cpu())
                self.all_states.append(states[:,seq_idx,:].detach().cpu())
