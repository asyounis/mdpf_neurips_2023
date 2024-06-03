# Standard Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch Imports
import torch

# Project Imports
from evaluations.evaluation_base import *

class ProposalEvaluationBase(EvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, save_dir, device, seed)

        #Save the model
        self.model = model

        # Parse the evaluation parameters
        evaluation_params = experiment["evaluation_params"]
        self.number_to_render = evaluation_params["number_to_render"]
        self.number_of_particles = evaluation_params["number_of_particles"]
        self.render_particles = evaluation_params["render_particles"]

        if("number_of_particle_hidden_dims" in evaluation_params):
            self.number_of_particle_hidden_dims = evaluation_params["number_of_particle_hidden_dims"]
        else:
            self.number_of_particle_hidden_dims = 0


        # If the model outputs a KDE then lets get a KDE going
        if(self.model.outputs_kde()):   
            self.kde_params = evaluation_params["kde_params"]




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
            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 12))
            axes = axes.reshape(-1,)

            # Render each of the initialization
            for i in range(self.number_to_render):

                # Grab the axis we will be using for this plot
                # and enable formatting things we want for it
                ax = axes[i]

                # Get a sample of data
                data = self.evaluation_dataset[dataset_values[i]]

                # Unpack the data and move to the device
                observations = data["observations"]
                states = data["states"].to(self.device)

                # Add Batch Dims
                observations = observations.unsqueeze(0)
                states = states.unsqueeze(0)

                # We may or may not have actions
                if("actions" in data):
                    actions = data["actions"].to(self.device)
                    actions = actions.unsqueeze(0)
                else:
                    actions = None

                # Create the particle set
                particles = torch.tile(states[:,0,:].unsqueeze(1),[1,  self.number_of_particles, 1])

                # Add the number of dims we need to add, if we need to add any
                if(self.number_of_particle_hidden_dims != 0):

                    # create the hidden state
                    hidden_state = torch.zeros((particles.shape[0], particles.shape[1], self.number_of_particle_hidden_dims), device=particles.device)

                    # append the hidden state to the particle state
                    particles = torch.cat([particles, hidden_state], dim=-1)


                particles = self.model.particle_transformer.backward_tranform(particles)


                # Encode the particles
                encoded_particles = self.model.particle_encoder_for_particles_model(particles)
        
                if(actions is not None):
                    # Encode the actions
                    tiled_actions = torch.tile(actions[:,0,:].unsqueeze(1),[1,  self.number_of_particles, 1])
                    encoded_actions = self.model.action_encoder_model(tiled_actions)

                    # Concatenate the encoded particles and actions
                    all_encodings = torch.cat([encoded_particles, encoded_actions], dim=-1)

                else:
                    # If we haev no encoded actions then we just use the encoded particles
                    all_encodings = encoded_particles

                # Generate new particles
                new_particles, _ = self.model.proposal_model(particles, all_encodings, None)

                # Decode the particle 
                new_particles = self.model.particle_transformer.forward_tranform(new_particles)

                # Assume equally weighted particles for the KDE
                new_particle_weights = torch.ones(size=(1, self.number_of_particles), device=self.device)
                new_particle_weights = new_particle_weights / self.number_of_particles

                # Create the output dict
                output_dict = dict()
                output_dict["particles"] = new_particles
                output_dict["particle_weights"] = new_particle_weights

                # If we output a KDE 
                if(self.model.outputs_kde()):
                    # Compute the best bandwidth for the set of particles
                    bandwidths = self.model.weighted_bandwidth_predictor(new_particles.detach(), weights=new_particle_weights.detach())
                    output_dict["bandwidths"] = bandwidths

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