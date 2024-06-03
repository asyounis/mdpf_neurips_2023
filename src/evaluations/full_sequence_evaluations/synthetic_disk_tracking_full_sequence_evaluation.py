# Standard Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_video_base import *
from models.sequential_models import *

class SyntheticDiskTrackingFullSequenceEvaluation(FullSequenceEvaluationVideoBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)


    def get_rows_and_cols(self):
            
        rows = 3
        cols = 2

        if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):
            cols += 2

        return rows, cols


    def extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None

    def _convert_to_numpy(self, x):
        return x.detach().cpu().numpy()

    def render_frame(self, frame_number, data, output_dicts, axes):

        # For this one, reshape back into rows and cols
        rows, cols = self.get_rows_and_cols()
        axes = axes.reshape((rows, cols))

        # Extract stuff if we have them
        particle_weights = self.extract_if_present("particle_weights", output_dicts)

        # Draw the bounding boxes
        self.draw_bounding_boxes(frame_number, axes[0, 0], data, output_dicts, "State Output")

        # self.draw_table_info(frame_number, axes[1, 0], data, output_dicts, "State Output")

        # Render the state
        self.draw_xy_state(frame_number, axes[0, 1], data, output_dicts, "State Output Center")
        self.draw_width_height_state(frame_number, axes[1, 0], data, output_dicts, "State Output Width/Height")
        # # Render the angle
        # self.draw_angle_state(frame_number, axes[1,0], data, output_dicts, "State Angle Output")
        

        # Render the ESS if we have the weights
        if(particle_weights is not None):
            self.add_ess_to_plot(axes[2,0], particle_weights, "State Output")

        # If we output a KDE then plot the bandwidths so we can see them
        if(self.model.outputs_kde()):
            bandwidths = output_dicts["bandwidths"]
            self.plot_bandwidths(frame_number, axes[2, 1], bandwidths, "Bandwidths")

            # If we have a different bandwidth for resampling then we want to see that too
            if(self.model.decouple_bandwidths_for_resampling):
                resampling_bandwidths = output_dicts["resampling_bandwidths"]
                self.plot_bandwidths(frame_number, axes[2, 3], resampling_bandwidths, "Resampling Bandwidths")


        # If we are resampling then we also want to render the Resampling ESS
        if(self.model.decouple_weights_for_resampling):

            # Extract stuff if we have them
            resampling_particle_weights = self.extract_if_present("resampling_particle_weights", output_dicts)

            # If we have the weights then display the ESS
            if(resampling_particle_weights is not None):
                self.add_ess_to_plot(axes[2, 2], resampling_particle_weights, "Resampling")

        # If we have Different resampling KDE values then we should draw them
        if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):

            # Render the xy state
            self.draw_xy_state(frame_number, axes[0, 2], data, output_dicts, "Resampling Output Center", "resampling_")
            self.draw_width_height_state(frame_number, axes[1, 3], data, output_dicts, "Resampling Output Width/Height","resampling_")



    def draw_bounding_boxes(self,frame_number, ax, data, output_dicts, title):

        # Unpack
        observations = data["observations"].clone()
        particles = output_dicts["particles"].clone()
        particle_weights = output_dicts["particle_weights"].clone()

        max_weight = torch.max(particle_weights)

        # Draw the image
        image = observations[0, frame_number]
        image = torch.permute(image, (1, 2, 0))
        image = self._convert_to_numpy(image)
        ax.imshow(image)

        # Extract for this frame
        particles = particles[frame_number, 0]
        particle_weights = particle_weights[frame_number, 0]

        # resale to the image scale
        particles /= 10.0
        particles *= (observations.shape[-1]//2)

        # Shift so that we are within [0, image_size] instead of [-image_size//2, image_size//2]
        particles[..., :2] += (observations.shape[-1]//2)

        # Create bounding box for drawing
        widths = particles[..., 2]
        heights = particles[..., 3]
        llx = particles[...,0] - (widths/2.0)
        lly = particles[...,1] - (heights/2.0)

        # Convert everything to numpy
        widths = self._convert_to_numpy(widths)
        heights = self._convert_to_numpy(heights)
        llx = self._convert_to_numpy(llx)
        lly = self._convert_to_numpy(lly)

        # Compute the alpha value of the particle.  Most weighted particles are most opaque
        # max_weight = torch.max(particle_weights)
        # min_weight = torch.min(particle_weights)
        # particle_alpha_values = (particle_weights - min_weight) / (max_weight - min_weight)
        particle_alpha_values = (particle_weights / max_weight)


        # Apply a min alpha value
        min_alpha_value = 0.15
        particle_alpha_values[particle_alpha_values < min_alpha_value] = min_alpha_value

        # Draw bounding boxes on the images
        for pn in range(particles.shape[0]):

            alpha_value = particle_alpha_values[pn].item()

            # Draw a bounding box
            rect = patches.Rectangle((llx[pn], lly[pn]), widths[pn], heights[pn], linewidth=2.5, edgecolor="white", facecolor='none', alpha=alpha_value)
            ax.add_patch(rect)

        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

        # Set the title
        ax.set_title(title)

    def plot_bandwidths(self,frame_number, ax, bandwidths, title):

        # Extract the bandwidths
        bandwidths = bandwidths[:, 0]

        # Convert to numpy for plotting
        bandwidths = self._convert_to_numpy(bandwidths)

        # Plot
        for i in range(bandwidths.shape[-1]):
            ax.plot(bandwidths[:, i], label="Band. {}".format(i))    

        # Set the plot params
        ax.set_title(title)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Bandwidth")
        ax.legend()


    def draw_xy_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

        # Set the X and Y limits
        ax.set_xlim(self.evaluation_dataset.get_x_range())
        ax.set_ylim(self.evaluation_dataset.get_y_range())

        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Extract the data that we need
            particles = output_dicts["particles"].clone()

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

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            x_samples = samples[0, :, 0].cpu().numpy()
            y_samples = -samples[0, :, 1].cpu().numpy()
            ranges = [self.evaluation_dataset.get_x_range(), self.evaluation_dataset.get_y_range()]
            ax.hist2d(x_samples, y_samples, range=ranges, bins=200, cmap="Blues")


        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"].clone()
            particles = particles[frame_number]

            # Draw the particles
            x = self._convert_to_numpy(particles[0, :, 0])
            y = -self._convert_to_numpy(particles[0, :, 1])
            ax.scatter(x,y, color="black", s=1)

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
        states = data["states"].clone()

        # Determine the size of the true location circle
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        min_range = min(y_max-y_min, x_max-x_min)
        circle_size = min_range*0.025

        # Plot the true location
        true_x = states[0,frame_number,0].item()
        true_y = -states[0,frame_number,1].item()
        true_state_circle = plt.Circle((true_x, true_y), circle_size, color='red')
        ax.add_patch(true_state_circle)


        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
        ax.set_title(title)

        # Make it a square
        ax.set_aspect('equal', adjustable='box')



    def draw_width_height_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):


        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Extract the data that we need
            particles = output_dicts["particles"].clone()

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

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            width_samples = samples[0, :, 2].cpu().numpy()
            height_samples = samples[0, :, 3].cpu().numpy()
            # ranges = [[0, 10], [0, 10]]
            # ax.hist2d(width_samples, height_samples, range=ranges, bins=200, cmap="Blues", alpha=0.5)
            ax.hist2d(width_samples, height_samples, bins=200, cmap="Blues", alpha=1.0)



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"].clone()
            particles = particles[frame_number]

            # Draw the particles
            width = self._convert_to_numpy(particles[0, :, 2])
            height = self._convert_to_numpy(particles[0, :, 3])
            ax.scatter(width, height, color="black", s=1)

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
        states = data["states"].clone()

        # Determine the size of the true location circle
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        min_range = min(y_max-y_min, x_max-x_min)
        circle_size = min_range*0.025

        # Plot the true location
        true_width = states[0,frame_number,2].item()
        true_height = states[0,frame_number,3].item()
        true_state_circle = plt.Circle((true_width, true_height), circle_size, color='red', alpha=1.0)
        ax.add_patch(true_state_circle)


        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
        ax.set_title(title)

        # Make it a square
        ax.set_aspect('equal', adjustable='box')

    # def draw_table_info(self, frame_number, ax, data, output_dicts, title):

    #     # fig.patch.set_visible(False)
    #     ax.axis('off')
    #     ax.axis('tight')



    #     rows = []
    #     rows.append([1, 10])
    #     rows.append([22, 4])
    #     ax.table(cellText=rows, loc="center")



