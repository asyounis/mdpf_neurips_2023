# Standard Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch Imports
import torch

# Project Imports
from evaluations.evaluation_base import *

class InitilizerEvaluationBase(EvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, save_dir, device, seed)

        #Save the model
        self.model = model

        # Parse the evaluation parameters
        evaluation_params = experiment["evaluation_params"]
        self.number_to_render = evaluation_params["number_to_render"]
        self.number_of_particles = evaluation_params["number_of_particles"]
        self.render_particles = evaluation_params["render_particles"]

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
                observations = data["observations"].to(self.device)
                states = data["states"]#.to(self.device)

                # We only care for the first observation 
                observations = observations[0:1, ...]

                # transform the observations to the correct space
                transformed_observations = self.problem.observation_transformer.forward_tranform(observations)

                # Get the initial state
                output_dict = self.model.create_initial_dpf_state(states[0,:].unsqueeze(0), transformed_observations.unsqueeze(0), self.number_of_particles)

                # Render that experiment
                self.render_experiment(ax, output_dict, data)

        # Get rid of unnecessary white space
        fig.tight_layout()

        # plt.show()

        # Save the figure
        plt.savefig("{}/renderings.png".format(self.save_dir))

    def get_number_of_axis_per_experiment(self):
        raise NotImplemented

    def render_experiment(self, ax, output_dict, data):
        raise NotImplemented