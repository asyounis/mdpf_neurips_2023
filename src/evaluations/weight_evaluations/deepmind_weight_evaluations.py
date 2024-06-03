# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.weight_evaluations.weight_evaluation_base import *
from models.sequential_models import *


class DeepmindMazeWeightEvaluation(WeightEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def get_number_of_axis_per_experiment(self):
        return 1

    def render_experiment(self, ax, output_dict, observations, states):

        # Get the map id
        map_id = output_dict["map_id"].item()

        states = self.evaluation_dataset.scale_data_up(states, map_id)

        # Compute the base size that we will use scaling all the rendered objects
        x_range = self.evaluation_dataset.get_x_range(map_id)[1] - self.evaluation_dataset.get_x_range(map_id)[0]
        y_range = self.evaluation_dataset.get_y_range(map_id)[1] - self.evaluation_dataset.get_y_range(map_id)[0]
        max_range = max(x_range, y_range)
        rendered_object_base_size = max_range / 40.0

        # Set the X and Y limits
        # ax.set_xlim([-15, 15])
        # ax.set_ylim([-15, 15])
        ax.set_xlim(self.evaluation_dataset.get_x_range(map_id))
        ax.set_ylim(self.evaluation_dataset.get_y_range(map_id))

        # Draw KDE
        self.draw_kde(ax, output_dict, map_id)


        # Plot the true starting location
        starting_x = states[0,0].item()
        starting_y = states[0,1].item()
        true_state_circle = plt.Circle((starting_x, starting_y), 0.25*rendered_object_base_size, color='red')
        ax.add_patch(true_state_circle)
        dy = torch.sin(states[0,2]).item() * 1.0 * rendered_object_base_size
        dx = torch.cos(states[0,2]).item() * 1.0 * rendered_object_base_size
        ax.arrow(starting_x, starting_y, dx, dy, color='red', width=0.15*rendered_object_base_size)

        # Draw the particles
        particles = output_dict["particles"]
        particles = self.evaluation_dataset.scale_data_up(particles, map_id)
        x = particles[0, :, 0].cpu().numpy()
        y = particles[0, :, 1].cpu().numpy()
        ax.scatter(x,y, color="black", s=1)

        # Draw the walls
        walls = self.evaluation_dataset.get_walls(map_id)
        ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="black", linewidth=3)


        # Add the ESS to the plot
        ess = output_dict["ess"][0].item()
        ax.text(0.5, 0.95, "ESS: {:0.2f}".format(ess), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    def draw_kde(self, ax, output_dict, map_id):

        # Extract the particles and weights
        particles = output_dict["particles"]
        particle_weights = output_dict["particle_weights"]

        # Get the bandwidth
        manual_bandwidth = self.kde_rendering_params["manual_bandwidth"]
        if(isinstance(manual_bandwidth, list)):

            # Make sure they specified the correct number of bandwidths
            assert(len(manual_bandwidth) == particles.shape[-1])

            # Create the bandwidth array
            bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1]), device=particles.device)
            for i in range(len(manual_bandwidth)):
                bandwidths[...,i] = manual_bandwidth[i]

        elif(manual_bandwidth is not None):
            bandwidths = torch.full(size=(particles.shape[0], particles.shape[-1]), fill_value=manual_bandwidth, device=particles.device)

        # Use a KDE to generate samples that we will use in the histogram
        kde = KernelDensityEstimator(self.kde_rendering_params["kde_params"], particles, particle_weights, bandwidths)
        samples = kde.sample((100000, ))

        # # Scale the samples to the correct domain
        samples = self.evaluation_dataset.scale_data_up(samples, map_id)

        # Draw the probability distribution of where we think we should put mass
        x_samples = samples[0, :, 0].cpu().numpy()
        y_samples = samples[0, :, 1].cpu().numpy()
        ranges = [self.evaluation_dataset.get_x_range(map_id), self.evaluation_dataset.get_y_range(map_id)]
        # ranges = [[-15, 15],[-15, 15]]
        ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")
