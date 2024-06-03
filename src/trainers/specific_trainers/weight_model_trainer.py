
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
from kernel_density_estimation.epanechnikov import *

# Project Imports
from trainers.trainer import *
from problems.problem_base import *


class ParticleCloudGenerator:
    def __init__(self, particle_cloud_params):

        # Extract the number of particles we will have for the particle cloud
        self.number_of_particles = particle_cloud_params["number_of_particles"]

        # We will need to get a center of a distribution so lets use some distribution
        self.cloud_center_distribution_params = particle_cloud_params["cloud_center_distribution_params"]

    def create_cloud(self, states):
        raise NotImplemented

    def sample_from_dist(self, states, distribution_params, number_of_samples):

        # The particle cloud structure that we will be using
        samples = torch.zeros((states.shape[0], number_of_samples, states.shape[1]), device=states.device)

        # Create the random cloud
        all_dims = distribution_params["dims"]
        for d in all_dims:

            # The parameters of the distribution
            distribution_type = all_dims[d]["distribution_type"]
            bandwidth = all_dims[d]["bandwidth"]

            # Create the appropriate distribution
            if(distribution_type == "Normal"):
                dist = D.Normal(loc=states[..., d], scale=bandwidth, validate_args=True)

            elif(distribution_type == "Von_Mises"):
                bandwidth = 1.0 / (bandwidth + 1e-8)
                dist = VonMisesFullDist(loc=states[..., d], concentration=bandwidth, validate_args=True)
            
            elif(distribution_type == "Epanechnikov"):
                dist = Epanechnikov(loc=states[..., d], bandwidth=bandwidth)

            else:
                assert(False)

            samples[:, :, d] = torch.permute(dist.sample((number_of_samples,)),(1, 0))


        return samples

class DistributionCloudGenerator(ParticleCloudGenerator):
    def __init__(self, particle_cloud_params):
        super().__init__(particle_cloud_params)

        # We will create a cloud using a distribution so extract the distribution parameters
        self.cloud_generating_distribution_params = particle_cloud_params["cloud_generating_distribution_params"]

    def create_cloud(self, states):
        
        # Generate the new cloud center
        cloud_center = self.sample_from_dist(states, self.cloud_center_distribution_params, 1).squeeze(1)

        # Generate the particle cloud
        particle_cloud = self.sample_from_dist(cloud_center, self.cloud_generating_distribution_params, self.number_of_particles)

        return particle_cloud


class WeightModelTrainer(Trainer):
    def __init__(self, model, problem, params, save_dir, device):
        super().__init__(params, model, problem, save_dir, device)

        # Extract the parameters we will use for the generating the particle cloud that we are going to weight
        particle_cloud_params = self.training_params["particle_cloud_params"]
        particle_cloud_generator_type = particle_cloud_params["particle_cloud_generator_type"]

        # Create the generator
        if(particle_cloud_generator_type == "Distribution"):
            self.particle_cloud_generator = DistributionCloudGenerator(particle_cloud_params)
        else:
            print("Unknown particle cloud generator type \"{}\"".format(particle_cloud_generator_type))
            assert(False)





    def get_training_type(self):
        return "weight"

    def do_forward_pass(self, data, dataset, epoch):

        # we might have a map, if we do then unpack it and move it to the correct device
        if("world_map" in data):
            world_map = data["world_map"]
            if(world_map is not None):
                world_map = world_map.to(self.device, non_blocking=True)
        else:
            world_map = None

        # Get all the states and observations that have true mask values
        states, observations, world_map = self.get_true_masked_data(data)

        # Move everything to the GPU
        observations = observations.to(self.device, non_blocking=True)
        states = states.to(self.device, non_blocking=True)

        # transform the observation
        observations = self.problem.observation_transformer.forward_tranform(observations)

        # Create the particle clouds
        particle_cloud = self.particle_cloud_generator.create_cloud(states)

        # Scale the particles
        particle_cloud = self.main_model.particle_transformer.downscale(particle_cloud)

        # Transform the particle set
        particle_cloud_transformed = self.main_model.particle_transformer.backward_tranform(particle_cloud)

        # Create a dictionary for the weight model inputs
        weight_model_processor_input_dict = dict()
        weight_model_processor_input_dict["observation"] = observations 
        weight_model_processor_input_dict["next_observation"] = None
        weight_model_processor_input_dict["internal_particles"] = particle_cloud_transformed 
        weight_model_processor_input_dict["particles"] = particle_cloud
        weight_model_processor_input_dict["old_particles"] = None 
        weight_model_processor_input_dict["reference_patch"] = None
        weight_model_processor_input_dict["world_map"] = world_map

        # Get the pre-processed inputs for the weights model
        weight_model_input = self.main_model.weight_model.input_processor.process(weight_model_processor_input_dict)

        # Weigh the particles
        particle_weights = self.main_model.weight_model(weight_model_input)

        # Normalize the weights
        particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)

        # Pack the output dict so we can use the loss functions
        output_dict = dict()
        output_dict["particles"] = self.main_model.particle_transformer.upscale(particle_cloud)
        output_dict["particle_weights"] = particle_weights

        # Compute the loss           
        loss = self.loss_function.compute_loss(output_dict, states)

        # Aggregate the loss via mean
        loss = torch.mean(loss)

        # Return the final loss and the "batch size" that was used to compute the loss 
        return loss, states.shape[0]


    def get_true_masked_data(self, data):

        # Unpack the data 
        states = data["states"]
        observations = data["observations"]

        # we might have a map, if we do then unpack it and move it to the correct device
        if("world_map" in data):
            world_map = data["world_map"]
            if(world_map is not None):
                world_map = world_map.to(self.device, non_blocking=True)
        else:
            world_map = None


        # Extract some statistics
        batch_size = states.shape[0]
        subsequence_length = states.shape[1]

        # Extract the ground truth mask if we have one
        if("ground_truth_mask" in data):
            ground_truth_mask = data["ground_truth_mask"]

            # We only support 1 type of mask for now
            true_count = 0
            for seq_idx in range(subsequence_length):

                if(ground_truth_mask[0, seq_idx].item() == True):
                    assert(torch.sum(ground_truth_mask[:, seq_idx]).item() == ground_truth_mask.shape[0])
                    true_count += 1

            # Make sure we have at least 1 true mask value otherwise WTF
            assert(true_count > 0)

        else:

            # If we dont have a mask them make one real quick where all the steps are true
            ground_truth_mask = torch.full(size=(states.shape[0],states.shape[1]), fill_value=True)

        # Get all the states that have a true ground true states, all the other states are invalid
        valid_states = []
        valid_observations = []
        valid_world_maps = []
        for seq_idx in range(subsequence_length):
            if(ground_truth_mask[0, seq_idx].item() == False):
                continue

            # these timesteps have a true groundt truth mask value
            valid_states.append(states[:, seq_idx])
            valid_observations.append(observations[:, seq_idx])

            if(world_map is not None):
                valid_world_maps.append(world_map)


        # Put into 1 big batch!!
        valid_states = torch.vstack(valid_states)
        valid_observations = torch.vstack(valid_observations)

        if(world_map is not None):
            valid_world_maps = torch.vstack(valid_world_maps)
        else:
            valid_world_maps = None

        return valid_states, valid_observations, valid_world_maps














