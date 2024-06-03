# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.proposal_evaluations.proposal_evaluation_base import *
from models.sequential_models import *

class ToyProblemProposalEvaluations(ProposalEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def get_number_of_axis_per_experiment(self):
        return 1

    def render_experiment(self, ax, output_dict, observations, states):

        # Set the limits
        ax.set_xlim(self.evaluation_dataset.get_range())

        if(self.model.outputs_kde()):
            # Extract the data that we need
            particles = output_dict["particles"] 
            particle_weights = output_dict["particle_weights"] 
            bandwidths = output_dict["bandwidths"] 

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

            # Create a set set of test_points to evaluate the KDE on
            states_range = self.evaluation_dataset.get_range()
            test_points = torch.linspace(states_range[0], states_range[1], 100000).unsqueeze(0).unsqueeze(-1).to(particles.device)

            # Evaluate at the test points
            probs = torch.exp(kde.log_prob(test_points))
            ax.plot(test_points.cpu().squeeze(), probs.cpu().squeeze(), label="KDE")



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the particles
            particles = output_dict["particles"]

            # Draw the particles
            x = particles[0, :, 0].cpu().numpy()
            y = torch.zeros_like(particles[0, :, 0].cpu())
            ax.scatter(x,y, color="black", s=5, label="Particles")

        # Plot the true location
        starting_x = states[0,0,0].item()        
        ax.axvline(x=starting_x, color="red", label="Starting")

        # Plot the true location
        true_x = states[0,1,0].item()        
        ax.axvline(x=true_x, color="blue", label="True")

        # Set that we have a legend
        ax.legend()
