# Standard Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.distributions as D

# Project Imports
from evaluations.evaluation_base import *


# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *




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
                dist = Epanechnikov(states[...,d], bandwidth)

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




class WeightEvaluationBase(EvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, save_dir, device, seed)

        #Save the model
        self.model = model.to(device)

        # Parse the evaluation parameters
        evaluation_params = experiment["evaluation_params"]
        self.number_to_render = evaluation_params["number_to_render"]


        # Extract the parameters we will use for the generating the particle cloud that we are going to weight
        particle_cloud_params = evaluation_params["particle_cloud_params"]
        particle_cloud_generator_type = particle_cloud_params["particle_cloud_generator_type"]

        # Create the generator
        if(particle_cloud_generator_type == "Distribution"):
            self.particle_cloud_generator = DistributionCloudGenerator(particle_cloud_params)
        else:
            print("Unknown particle cloud generator type \"{}\"".format(particle_cloud_generator_type))
            assert(False)

        # Extract the KDE rendering params
        self.kde_rendering_params = evaluation_params["kde_rendering_params"]

    def run_evaluation(self):

        # We want to be fast and speedy
        with torch.no_grad():

            # Set the model to be eval mode for evaluation
            self.model.eval()

            # Create a list of shuffled indices for the dataset
            dataset_values = list(range(len(self.evaluation_dataset)))
            random.shuffle(dataset_values)

            # Figure out how many rows and cols to have
            cols = min(int(np.sqrt(self.number_to_render)), 4)
            rows = 1
            while((cols*rows < self.number_to_render)):
                rows += 1

            # Make the figure
            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(16, 12))
            axes = axes.reshape(-1,)

            # Render each of the initialization
            for i in range(self.number_to_render):

                # Grab the axis we will be using for this plot
                # and enable formatting things we want for it
                ax = axes[i]

                # Get a sample of data
                data = self.evaluation_dataset[dataset_values[i]]

                # we might have a map, if we do then unpack it and move it to the correct device
                if("world_map" in data):
                    assert(False)
                    # world_map = data["world_map"]
                    # if(world_map is not None):
                    #     world_map = world_map.to(self.device, non_blocking=True)
                else:
                    world_map = None

                if("map_id" in data):
                    map_id = data["map_id"]
                else:
                    map_id = None

                # Get all the states and observations that have true mask values
                states, observations = self.get_true_masked_data(data)

                # Get only 1 state/obs
                # states = states[0, ...].unsqueeze(0)
                # observations = observations[0, ...].unsqueeze(0)

                # Move everything to the GPU
                observations = observations.to(self.device, non_blocking=True)
                states = states.to(self.device, non_blocking=True)

                # Create the particle clouds
                particle_cloud = self.particle_cloud_generator.create_cloud(states)

                # Transform the particle set
                particle_cloud_transformed = self.model.particle_transformer.backward_tranform(particle_cloud)

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
                weight_model_input = self.model.weight_model.input_processor.process(weight_model_processor_input_dict)

                # Weigh the particles
                particle_weights = self.model.weight_model(weight_model_input)

                # Normalize the weights
                particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)

                # Compute the ESS
                ess = particle_weights**2
                ess = torch.sum(ess, dim=-1)
                ess = 1.0 / ess

                # Pack the output dict so we can use the loss functions
                output_dict = dict()
                output_dict["particles"] = particle_cloud
                output_dict["particle_weights"] = particle_weights
                output_dict["map_id"] = map_id
                output_dict["ess"] = ess

                # Render that experiment
                self.render_experiment(ax, output_dict, observations, states)

        # Get rid of unnecessary white space
        fig.tight_layout()

        # plt.show()

        # Save the figure
        plt.savefig("{}/renderings.png".format(self.save_dir))

    def get_number_of_axis_per_experiment(self):
        raise NotImplemented

    def render_experiment(self, ax, output_dict, observations, states):
        raise NotImplemented





    def get_true_masked_data(self, data):

        # Unpack the data 
        states = data["states"]
        observations = data["observations"]

        add_batch_dim = False
        if(len(states.shape) == 2):
            add_batch_dim = True

        if(add_batch_dim):
            states = states.unsqueeze(0)                
            observations = observations.unsqueeze(0)               


        # Extract some statistics
        batch_size = states.shape[0]
        subsequence_length = states.shape[1]

        # Extract the ground truth mask if we have one
        if("ground_truth_mask" in data):
            ground_truth_mask = data["ground_truth_mask"]

            if(add_batch_dim):
                 ground_truth_mask = ground_truth_mask.unsqueeze(0)                

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
        for seq_idx in range(subsequence_length):
            if(ground_truth_mask[0, seq_idx].item() == False):
                continue

            # these timesteps have a true groundt truth mask value
            valid_states.append(states[:, seq_idx])
            valid_observations.append(observations[:, seq_idx])

        # Put into 1 big batch!!
        valid_states = torch.vstack(valid_states)
        valid_observations = torch.vstack(valid_observations)

        return valid_states, valid_observations














