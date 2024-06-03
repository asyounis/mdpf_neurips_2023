# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_image_base import *
from models.sequential_models import *

class SimpleBrownianMotionTrackingFullSequenceEvaluation(FullSequenceEvaluationImageBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

        # Parse the evaluation parameters
        evaluation_params = experiment["evaluation_params"]

        # If we have a manual bandwidth set then we need use it
        if("manual_bandwidth" in evaluation_params):
            self.manual_bandwidth = evaluation_params["manual_bandwidth"]
        else:
            self.manual_bandwidth = None

    def get_rows_and_cols(self):
            
        rows = 2
        cols = 1

        # if(self.model.decouple_weights_for_resampling):

        #     # 1 - Resampling State 
        #     # 2 - Resampling ESS
        #     rows += 2

        return rows, cols


    def _extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None

    def _compute_y_lim(self, data, output_dicts):

        # Extract the data 
        states = data["states"]
        observations = data["observations"]
        particles = self._extract_if_present("particles", output_dicts)


        # Get the max and mins
        y_max = max(torch.max(states).item(), torch.max(observations).item())
        y_min = min(torch.min(states).item(), torch.min(observations).item())

        if(particles is not None):
            y_max = max(y_max, torch.max(particles).item())
            y_min = min(y_min, torch.min(particles).item())



        # Compute the Y axis range
        y_center = (y_max + y_min) / 2.0
        y_range = (y_max - y_min) * 1.1
        y_lims = (y_center - (y_range/2), y_center + (y_range/2))

        return y_lims

    def render_data(self, data, output_dicts, axes):

        # Get he y lims for this sequence
        y_lims = self._compute_y_lim(data, output_dicts)


        # Extract useful stats
        subsequence_length = data["states"].shape[1]

        # Generate the plotting steps (x axis) so that everything can be aligned in the plots
        plotting_steps = np.arange(0, subsequence_length) + 0.5



        self._render_problem(axes[0,0], data, plotting_steps, y_lims)
        self._render_pf_output(axes[1,0], data, plotting_steps, y_lims, output_dicts)
        
        # # axes[0,0].set_ylim(axis_min, axis_max)
        # axes[0,0].plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
        # axes[0,0].plot(plotting_steps, most_probable_states[:,0].cpu().numpy(), label="Most Probable KDE State", color="blue")
        # axes[0,0].plot(plotting_steps, most_probable_particles[:,0].cpu().numpy(), label="Most Particle", color="orange")
        # # axes[0,0].vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
        # axes[0,0].set_title("State Predictions")
        # axes[0,0].set_xlabel("Time-step")
        # axes[0,0].set_ylabel("State")
        # axes[0,0].legend()


    def _render_problem(self, ax, data, plotting_steps, y_lims):

        # Extract the data 
        states = data["states"].cpu().numpy()
        observations = data["observations"].cpu().numpy()
        kalman_filter_estimate = data["kalman_filter_estimate"][0].cpu().numpy()


        # Plot the KF estimate
        ax.plot(plotting_steps, kalman_filter_estimate[:, 0], color="green", label="Kalman Filter Mean")
        ax.fill_between(plotting_steps, kalman_filter_estimate[:, 0]+kalman_filter_estimate[:, 1], kalman_filter_estimate[:, 0]-kalman_filter_estimate[:, 1], facecolor='red', alpha=0.25)

        # Plot the lines
        ax.plot(plotting_steps, states[0, :, 0], label="True State", color="red")
        ax.plot(plotting_steps, observations[0, :, 0], label="Observations", color="blue")

        # Some formatting
        ax.set_title("Simple Brownian Motion Tracking Data")
        ax.set_xlabel("Time-step")
        ax.set_ylabel("State")
        ax.legend()


    def _render_pf_output(self, ax, data, plotting_steps, y_lims, output_dicts):

        # Extract the data 
        states = data["states"].cpu().numpy()

        # Extract stuff if we have them        
        particles = self._extract_if_present("particles", output_dicts)
        particle_weights = self._extract_if_present("particle_weights", output_dicts)
        bandwidths = self._extract_if_present("bandwidths", output_dicts)

        # Generate the image
        kde_image_data = self._create_kde_image(particles, particle_weights, bandwidths, y_lims)

        # Extract the single state solutions from the KDE
        most_probable_states, most_probable_particles = self._extract_single_state_solution(particles, particle_weights, bandwidths, y_lims)

        # Render the image
        ax.imshow(kde_image_data, extent=[0, kde_image_data.shape[1], y_lims[0], y_lims[1]], aspect="auto", interpolation="none",cmap=plt.get_cmap("Greys"))

        # Plot the lines
        ax.plot(plotting_steps, states[0, :, 0], label="True State", color="red")
        ax.plot(plotting_steps, most_probable_states[:,0].cpu().numpy(), label="Most Probable KDE State", color="blue")
        ax.plot(plotting_steps, most_probable_particles[:,0].cpu().numpy(), label="Highest WeightedParticle", color="orange")


        # Some formatting
        ax.set_title("Particle Filter Estimate ")
        ax.set_xlabel("Time-step")
        ax.set_ylabel("State")
        ax.legend()



    def _extract_single_state_solution(self, particles, particle_weights, bandwidths, y_lims):
        
        # Extract some stats
        subsequence_length = particles.shape[0]

        if(bandwidths is not None):


            # Create some test points that we will use for creating the image
            test_points = torch.linspace(y_lims[0], y_lims[1], 100000).unsqueeze(0).unsqueeze(-1).to(particles.device)

            # Get the probability of all the the test points
            most_probable_states = []
            most_probable_particles = []
            for i in range(subsequence_length):

                # Create the KDE
                kde = KernelDensityEstimator(self.kde_params, particles[i], particle_weights[i], bandwidths[i])

                # Get the probability of the test points
                log_probs = kde.log_prob(test_points)

                # Extract the most probable state
                most_probable_state_index = torch.argmax(log_probs)
                most_probable_state = test_points[0, most_probable_state_index, 0]
                most_probable_states.append(most_probable_state)

                # Extract the most probable particle
                most_probable_particle_index = torch.argmax(particle_weights[i, 0])
                most_probable_particle = particles[i, 0, most_probable_particle_index, 0]
                most_probable_particles.append(most_probable_particle)

            # Stack and add the "state dim" dim back
            most_probable_states = torch.stack(most_probable_states).unsqueeze(-1)
            most_probable_particles = torch.stack(most_probable_particles).unsqueeze(-1)

            return most_probable_states, most_probable_particles

        else:

            # Get the probability of all the the test points
            most_probable_states = []
            most_probable_particles = []
            for i in range(subsequence_length):

                # Extract the most probable state
                most_probable_state_index = torch.argmax(particle_weights[i])
                most_probable_state = particles[i][0,most_probable_state_index, 0]
                most_probable_states.append(most_probable_state)

                # Extract the most probable particle
                most_probable_particle_index = torch.argmax(particle_weights[i, 0])
                most_probable_particle = particles[i, 0, most_probable_particle_index, 0]
                most_probable_particles.append(most_probable_particle)

            # Stack and add the "state dim" dim back
            most_probable_states = torch.stack(most_probable_states).unsqueeze(-1)
            most_probable_particles = torch.stack(most_probable_particles).unsqueeze(-1)

            return most_probable_states, most_probable_particles





    def _create_kde_image(self, particles, particle_weights, bandwidths, y_lims):

        # Extract some stats
        subsequence_length = particles.shape[0]

        # Create some test points that we will use for creating the image
        test_points = torch.linspace(y_lims[0], y_lims[1], 100000).unsqueeze(0).unsqueeze(-1).to(particles.device)

        # Get the probability of all the the test points
        all_probs = []
        for i in range(subsequence_length):

            # Create the KDE
            kde = KernelDensityEstimator(self.kde_params, particles[i], particle_weights[i], bandwidths[i])

            # Get the probability of the test points
            probs = torch.exp(kde.log_prob(test_points))
            all_probs.append(probs)

        # Make it into 1 big tensor and remove the batch dim
        all_probs = torch.stack(all_probs)
        all_probs = all_probs.squeeze(1)

        # Generate the likelihood Image
        kde_image_data = all_probs
        mins,_ = torch.min(kde_image_data,dim=-1, keepdim=True)
        maxs,_ = torch.max(kde_image_data,dim=-1, keepdim=True)
        kde_image_data = kde_image_data - mins
        kde_image_data = kde_image_data / (maxs-mins)
        kde_image_data = kde_image_data.cpu().numpy().transpose()
        kde_image_data = np.flip(kde_image_data, axis=0)


        return kde_image_data

