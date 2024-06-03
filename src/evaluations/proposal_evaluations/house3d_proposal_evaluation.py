# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.proposal_evaluations.proposal_evaluation_base import *
from models.sequential_models import *

class House3DProposalEvaluations(ProposalEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def get_number_of_axis_per_experiment(self):
        return 1

    def render_experiment(self, ax, output_dict, observations, states):

        # Extract the start and true locations
        starting_x = states[0,0,0].item()
        starting_y = states[0,0,1].item()
        true_x = states[0,1,0].item()
        true_y = states[0,1,1].item()


        # Compute the max and min so we can center the plot around that
        max_x = max(starting_x, true_x)
        min_x = min(starting_x, true_x)
        max_y = max(starting_y, true_y)
        min_y = min(starting_y, true_y)



        # If we have particles then use them to get the max range
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the particles
            particles = output_dict["particles"]
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            max_x = max(max_x, np.max(x))
            min_x = min(min_x, np.min(x))
            max_y = max(max_y, np.max(y))
            min_y = min(min_y, np.min(y))

        max_range = max(max_x-min_x, max_y-min_y)
        max_range = max_range * 3
        
        if(max_range < 20.0):
            max_range = 20.0

        plotting_scale_factor = max_range * 0.05

        # Set the X and Y limits
        x_range_min = starting_x - max_range/2
        x_range_max = x_range_min + max_range
        y_range_min = starting_y - max_range/2
        y_range_max = y_range_min + max_range
        ax.set_xlim([x_range_min, x_range_max])
        ax.set_ylim([y_range_min, y_range_max])


        # Plot the true starting location
        true_state_circle = plt.Circle((starting_x, starting_y), plotting_scale_factor * 0.5, color='red', zorder=-1)
        ax.add_patch(true_state_circle)

        # Plot the true next location
        true_state_circle = plt.Circle((true_x, true_y), plotting_scale_factor * 0.4, color='blue', zorder=-1)
        ax.add_patch(true_state_circle)


        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the particles
            particles = output_dict["particles"]

            # Draw the particles
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            ax.scatter(x,y, color="green", s=5, zorder=10)
