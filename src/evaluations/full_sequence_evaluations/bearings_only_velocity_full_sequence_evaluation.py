# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_video_base import *
from models.sequential_models import *

class BearingsOnlyVelocityFullSequenceEvaluation(FullSequenceEvaluationVideoBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)


    def get_rows_and_cols(self):
            
        # 1 - State 
        # 2 - State angle
        # 3 - ESS
        rows = 3
        cols = 1

        if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):
            cols += 1

        return rows, cols


    def extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None

    def render_frame(self, frame_number, data, output_dicts, axes):

        # For this one, reshape back into rows and cols
        rows, cols = self.get_rows_and_cols()
        axes = axes.reshape((rows, cols))

        # Extract stuff if we have them
        particle_weights = self.extract_if_present("particle_weights", output_dicts)

        # Render the xy state
        self.draw_xy_state(frame_number, axes[0, 0], data, output_dicts, "State Output")

        # # Render the angle
        # self.draw_velocity_state(frame_number, axes[1,0], data, output_dicts, "Velocity Output")
        

        # # Render the ESS if we have the weights
        # if(particle_weights is not None):
        #     self.add_ess_to_plot(axes[2,0], particle_weights, "State Output")

        # # If we are resampling then we also want to render the Resampling ESS
        # if(self.model.decouple_weights_for_resampling):

        #     # Render the xy state
        #     self.draw_xy_state(frame_number, axes[0,1], data, output_dicts, "Resampling","resampling_")

        #     # Render the velocities
        #     self.draw_velocity_state(frame_number, axes[1,1], data, output_dicts, "Velocity Resampling", "resampling_")

        #     # Extract stuff if we have them
        #     resampling_particle_weights = self.extract_if_present("resampling_particle_weights", output_dicts)

        #     # If we have the weights then display the ESS
        #     if(resampling_particle_weights is not None):
        #         self.add_ess_to_plot(axes[2, 1], resampling_particle_weights, "Resampling")


    def draw_xy_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

        rendered_object_base_size = self.evaluation_dataset.get_x_range()[1] - self.evaluation_dataset.get_x_range()[0]
        rendered_object_base_size *= 0.05

        # Set the X and Y limits
        ax.set_xlim(self.evaluation_dataset.get_x_range())
        ax.set_ylim(self.evaluation_dataset.get_y_range())

        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Extract the data that we need
            particles = output_dicts["particles"]

            # Extract the particle weights with the prefix, otherwise just use the other weights
            if("{}particle_weights".format(key_name_prefix) in output_dicts):
                particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
            else:
                particle_weights = output_dicts["particle_weights"] 

            # If we are outputting a KDE then we have a bandwidth
            if(self.model.outputs_kde()):
                # Extract the bandwidths with the prefix, otherwise just use the other bandwidth
                if("{}bandwidths".format(key_name_prefix) in output_dicts):
                    bandwidths = output_dicts["{}bandwidths".format(key_name_prefix)] 
                else:
                    bandwidths = output_dicts["bandwidths"] 

            elif(self.use_manual_bandwidth and (self.manual_bandwidth is not None)):

                # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
                if(isinstance(self.manual_bandwidth, list)):

                    # Make sure they specified the correct number of bandwidths
                    assert(len(self.manual_bandwidth) == particles.shape[-1])

                    # Create the bandwidth array
                    bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), device=particles.device)
                    for i in range(len(self.manual_bandwidth)):
                        bandwidths[...,i] = self.manual_bandwidth[i]

                elif(self.manual_bandwidth is not None):
                    bandwidths = torch.full(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)


            else:
                print("Need a bandwidth from somewhere?!?!?!")
                assert(False)


            # Extract data we need for the frame number we need
            particles = particles[frame_number]
            particle_weights = particle_weights[frame_number]
            bandwidths = bandwidths[frame_number]

            # Add a text label Saying what the bandwidths ares
            text_str = "Bandwidths: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(bandwidths.squeeze()[0], bandwidths.squeeze()[1], bandwidths.squeeze()[2], bandwidths.squeeze()[3])
            ax.text(0.55, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            x_samples = samples[0, :, 0].cpu().numpy()
            y_samples = samples[0, :, 1].cpu().numpy()
            ranges = [self.evaluation_dataset.get_x_range(), self.evaluation_dataset.get_y_range()]
            ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"]
            particles = particles[frame_number]

            # Draw the particles
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            ax.scatter(x,y, color="black", s=1)

        if(self.model.outputs_single_solution()):

            # Extract the predicted state
            predicted_state = output_dicts["predicted_state"]

            print(predicted_state[frame_number,0,0])

            # Plot the true location
            y = predicted_state[frame_number,0,0,1].item()
            x = predicted_state[frame_number,0,0,0].item()
            true_state_circle = plt.Circle((x, y), 0.25 * rendered_object_base_size, color='blue')
            ax.add_patch(true_state_circle)
            dy = torch.sin(predicted_state[frame_number,0,0,2]).item() * 1.0 * rendered_object_base_size
            dx = torch.cos(predicted_state[frame_number,0,0,2]).item() * 1.0 * rendered_object_base_size
            ax.arrow(x, y, dx, dy, color='blue', width=0.15)


        # Extract the data we need for rendering
        observations = data["observations"]
        states = data["states"]

        # Transform the observations
        transformed_observation = self.problem.observation_transformer.forward_tranform(observations)

        # # Draw the sensor information 
        # sensors = self.evaluation_dataset.get_sensors()
        # for i, sensor in enumerate(sensors):

        #     # The sensor location
        #     position = sensor.get_position()
        #     x = position[0].item()
        #     y = position[1].item()
        #     ax.add_patch(plt.Circle((x, y), 0.5, color='green'))

        #     # The sensor observation
        #     sensor_obs_y = transformed_observation[frame_number, i].item() * 20
        #     sensor_obs_x = transformed_observation[frame_number, int(transformed_observation.shape[-1] / 2) + i].item() * 20
        #     x_values = [x, sensor_obs_x]
        #     y_values = [y, sensor_obs_y]
        #     ax.plot(x_values, y_values, color="green")


        # Plot the true location
        true_x = states[0, frame_number,0].item()
        true_y = states[0, frame_number,1].item()
        true_state_circle = plt.Circle((true_x, true_y), 0.25*rendered_object_base_size, color='red')
        ax.add_patch(true_state_circle)
        dy = states[0, frame_number,3].item() * 1.0 * rendered_object_base_size
        dx = states[0, frame_number,2].item() * 1.0 * rendered_object_base_size
        ax.arrow(true_x, true_y, 0, dy, color='red', width=0.15*rendered_object_base_size)
        ax.arrow(true_x, true_y, dx, 0, color='red', width=0.15*rendered_object_base_size)

        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
        ax.set_title(title)









    def draw_velocity_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

        # Set the X and Y limits
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Extract the data that we need
            particles = output_dicts["particles"]

            # Extract the particle weights with the prefix, otherwise just use the other weights
            if("{}particle_weights".format(key_name_prefix) in output_dicts):
                particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
            else:
                particle_weights = output_dicts["particle_weights"] 

            # If we are outputting a KDE then we have a bandwidth
            if(self.model.outputs_kde()):
                # Extract the bandwidths with the prefix, otherwise just use the other bandwidth
                if("{}bandwidths".format(key_name_prefix) in output_dicts):
                    bandwidths = output_dicts["{}bandwidths".format(key_name_prefix)] 
                else:
                    bandwidths = output_dicts["bandwidths"] 

            elif(self.use_manual_bandwidth and (self.manual_bandwidth is not None)):

                # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
                if(isinstance(self.manual_bandwidth, list)):

                    # Make sure they specified the correct number of bandwidths
                    assert(len(self.manual_bandwidth) == particles.shape[-1])

                    # Create the bandwidth array
                    bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), device=particles.device)
                    for i in range(len(self.manual_bandwidth)):
                        bandwidths[...,i] = self.manual_bandwidth[i]

                elif(self.manual_bandwidth is not None):
                    bandwidths = torch.full(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)


            else:
                print("Need a bandwidth from somewhere?!?!?!")
                assert(False)


            # Extract data we need for the frame number we need
            particles = particles[frame_number]
            particle_weights = particle_weights[frame_number]
            bandwidths = bandwidths[frame_number]

            # Add a text label Saying what the bandwidths ares
            text_str = "Bandwidths: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(bandwidths.squeeze()[0], bandwidths.squeeze()[1], bandwidths.squeeze()[2], bandwidths.squeeze()[3])
            ax.text(0.55, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            vx_samples = samples[0, :, 2].cpu().numpy()
            vy_samples = samples[0, :, 3].cpu().numpy()
            ranges = [[-3, 3], [-3, 3]]
            ax.hist2d(vx_samples, vy_samples, range=ranges, bins=100, cmap="Blues")



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"]
            particles = particles[frame_number]

            # Draw the particles
            vx = particles[0, :, 2].cpu().numpy()
            vy = particles[0, :, 3].cpu().numpy()
            ax.scatter(vx, vy, color="black", s=1)

        # if(self.model.outputs_single_solution()):

        #     # Extract the predicted state
        #     predicted_state = output_dicts["predicted_state"]

        #     print(predicted_state[frame_number,0,0])

        #     # Plot the true location
        #     y = predicted_state[frame_number,0,0,1].item()
        #     x = predicted_state[frame_number,0,0,0].item()
        #     true_state_circle = plt.Circle((x, y), 0.25, color='blue')
        #     ax.add_patch(true_state_circle)
        #     dy = torch.sin(predicted_state[frame_number,0,0,2]).item() * 1.0
        #     dx = torch.cos(predicted_state[frame_number,0,0,2]).item() * 1.0
        #     ax.arrow(x, y, dx, dy, color='blue', width=0.15)


        # Extract the data we need for rendering
        states = data["states"]

        # Plot the true location
        true_vx = states[0, frame_number,2].item()
        true_vy = states[0, frame_number,3].item()
        true_state_circle = plt.Circle((true_vx, true_vy), 0.1, color='red')
        ax.add_patch(true_state_circle)


        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
        ax.set_title(title)





