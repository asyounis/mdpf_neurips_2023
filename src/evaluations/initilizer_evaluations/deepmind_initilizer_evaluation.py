# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.initilizer_evaluations.initilizer_evaluation_base import *
from models.sequential_models import *

class DeepMindInitilizerEvaluations(InitilizerEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def get_number_of_axis_per_experiment(self):
        return 1

    def render_experiment(self, ax, output_dict, data):

        # Extract the map ID
        map_id = data["map_id"].item()

        # Extract the states and scale them to the correct scale
        states = data["states"]
        states = self.evaluation_dataset.scale_data_up(states, map_id)

        # Set the X and Y limits
        ax.set_xlim(self.evaluation_dataset.get_x_range(map_id))
        ax.set_ylim(self.evaluation_dataset.get_y_range(map_id))

        # Draw the walls
        walls = self.evaluation_dataset.get_walls(map_id)
        ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="black", linewidth=3)

        # Compute the base size that we will use scaling all the rendered objects
        x_range = self.evaluation_dataset.get_x_range(map_id)[1] - self.evaluation_dataset.get_x_range(map_id)[0]
        y_range = self.evaluation_dataset.get_y_range(map_id)[1] - self.evaluation_dataset.get_y_range(map_id)[0]
        max_range = max(x_range, y_range)
        rendered_object_base_size = max_range / 40.0

        if(self.model.outputs_kde()):
            # Extract the data that we need
            particles = output_dict["particles"] 
            particle_weights = output_dict["particle_weights"] 
            bandwidths = output_dict["bandwidths"] 

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            x_samples = samples[0, :, 0].cpu().numpy()
            y_samples = samples[0, :, 1].cpu().numpy()
            ranges = [self.evaluation_dataset.get_x_range(map_id), self.evaluation_dataset.get_y_range(map_id)]
            ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")


        # Plot the true location    
        true_x = states[0, 0].item()
        true_y = states[0, 1].item()
        true_state_circle = plt.Circle((true_x, true_y), 0.25*rendered_object_base_size, color='red')
        ax.add_patch(true_state_circle)
        dy = torch.sin(states[0, 2]).item() * 1.0*rendered_object_base_size
        dx = torch.cos(states[0, 2]).item() * 1.0*rendered_object_base_size
        ax.arrow(true_x, true_y, dx, dy, color='red', width=0.15*rendered_object_base_size)



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dict["particles"]

            # Scale them up
            particles = self.evaluation_dataset.scale_data_up(particles, map_id)

            # Draw the particles
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            ax.scatter(x,y, color="black", s=0.25*rendered_object_base_size)
