# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.initilizer_evaluations.initilizer_evaluation_base import *
from models.sequential_models import *

class BearingsOnlyInitilizerEvaluations(InitilizerEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def get_number_of_axis_per_experiment(self):
        return 1

    def render_experiment(self, ax, output_dict, data):

        # Unpack the data
        states = data["states"]
        observations = data["observations"]


        # Transform Observations
        transformed_observations = self.problem.observation_transformer.forward_tranform(observations)

        # Set the X and Y limits
        ax.set_xlim(self.evaluation_dataset.get_x_range())
        ax.set_ylim(self.evaluation_dataset.get_y_range())


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
            ranges = [self.evaluation_dataset.get_x_range(), self.evaluation_dataset.get_y_range()]
            ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")


        # Draw the sensor information 
        sensors = self.evaluation_dataset.get_sensors()
        for i, sensor in enumerate(sensors):

            # The sensor location
            position = sensor.get_position()
            x = position[0].item()
            y = position[1].item()
            ax.add_patch(plt.Circle((x, y), 0.5, color='green'))

            # The sensor observation
            sensor_obs_y = transformed_observations[0, i].item() * 20
            sensor_obs_x = transformed_observations[0, int(transformed_observations.shape[-1] / 2) + i].item() * 20
            x_values = [x, sensor_obs_x]
            y_values = [y, sensor_obs_y]
            ax.plot(x_values, y_values, color="green")


        # Plot the true location
        true_x = states[0,0].item()
        true_y = states[0,1].item()
        true_state_circle = plt.Circle((true_x, true_y), 0.5, color='red')
        ax.add_patch(true_state_circle)



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the particles
            particles = output_dict["particles"]

            # Draw the particles
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            ax.scatter(x,y, color="black", s=1)

