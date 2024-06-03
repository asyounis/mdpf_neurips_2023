# Standard Imports
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import cv2
import seaborn as sns
import copy 
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patheffects as patheffects


# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_video_base import *
from models.sequential_models import *

class House3DFullSequenceEvaluation(FullSequenceEvaluationVideoBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

        self.fig = None
        self.axes_dict = None

        # Check if we are decoupling posterior and resampling
        if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):
            self.decouple_posterior_and_resampling = True
        else:
            self.decouple_posterior_and_resampling = False


    def manually_manage_figure(self):
        return True

    def get_figure(self):

        # Figure out how many rows and cols we need
        rows = 4
        cols = 4

        # Make the figure
        # fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(4*cols, 3*rows), gridspec_kw={'width_ratios': [1, 1, 2]})
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols, 2.5*rows), gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5]})

        # Get the gridspec of the axis so we can use that to create larger subplots 
        gridspec = axes[0, 0].get_gridspec()

        # Put the axis into the dict
        self.axes_dict = dict()

        # Helper funcs
        def remove_axis(ax_to_remove):
            ax_flat = ax_to_remove.reshape(-1,)
            for ax in ax_flat:
                ax.remove()

        #################################################
        # Posterior Plots
        #################################################
        remove_axis(axes[0:2, 0:2])
        self.axes_dict["posterior_xy"] = fig.add_subplot(gridspec[0:2, 0:2])

        remove_axis(axes[0:2, 2:4])
        self.axes_dict["zoomed_posterior_xy"] = fig.add_subplot(gridspec[0:2, 2:4])

        # remove_axis(axes[2, 0:2])
        # self.axes_dict["posterior_angle"] = fig.add_subplot(gridspec[2, 0:2])

        remove_axis(axes[2:4, 0:2])
        self.axes_dict["posterior_angle"] = fig.add_subplot(gridspec[2:4, 0:2])

        remove_axis(axes[2:4, 2:4])
        self.axes_dict["observation"] = fig.add_subplot(gridspec[2:4, 2:4])


        # remove_axis(axes[3, 0:2])

        # The figure we are currently using
        self.fig = fig

        return self.fig


    def init_rendering(self, data, output_dicts, axes):
        # Clear all the axis for rendering in this frame
        for key in self.axes_dict.keys():
            self.axes_dict[key].clear()

        # Extract the data
        world_map = data["world_map"]
        particles = output_dicts["particles"]
        world_map_origional_size = data["world_map_origional_size"]

        # Get the data for this frame only
        particles = particles[0]
        world_map_origional_size = world_map_origional_size[0]

        # Get the width and the height of the world map
        world_width = world_map.shape[2]
        world_height = world_map.shape[1]

        # Create all the rendering objects that we need for rendering
        self.all_rendering_objects = dict()

        # All the rendering objects that get updated on each frame
        self.all_rendering_objects_that_get_updated = []

        ########################################################################################################################################################################
        ########################################################################################################################################################################
        # "posterior_xy" and zoomed version Axis Stuff
        ########################################################################################################################################################################
        ########################################################################################################################################################################
        
        def create_posterior_xy_rendering_objects(ax_name, is_zoomed, title):

            # Compute the render-able area
            if(is_zoomed):
                
                # How much to zoom
                ZOOM_FACTOR = 10

                # Compute the new widths
                new_world_width = world_width / ZOOM_FACTOR
                new_world_height = world_height / ZOOM_FACTOR

                # Compute the base size that we will use scaling all the rendered objects
                max_range = max(new_world_width, new_world_height)

                # Compute the base of the rendered object
                rendered_object_base_size = max_range / 20.0

            else:
                # Compute the base size that we will use scaling all the rendered objects
                max_range = max(world_width, world_height)

                # Compute the base of the rendered object
                rendered_object_base_size = max_range / 40.0


            rendering_objects = dict()
            rendering_objects_that_get_updated = []

            # The Full Path Stuff
            if(is_zoomed):
                self.draw_full_xy_path(self.axes_dict[ax_name], data, output_dicts, linewidth=0.75)
            else:
                self.draw_full_xy_path(self.axes_dict[ax_name], data, output_dicts, linewidth=2.0)

            # Draw the world map
            self.render_world_map(self.axes_dict[ax_name], world_map)

            # The XY stuff 
            if(self.model.outputs_kde()):
                rendering_objects["xy_state_kde_histogram"] = self.axes_dict[ax_name].imshow(np.random.uniform(size=(2, 2)), cmap="Blues", interpolation='none', norm=matplotlib.colors.LogNorm(), extent=[0, world_width, 0, world_height], origin="lower")
                rendering_objects_that_get_updated.append(rendering_objects["xy_state_kde_histogram"])

            # The state parts
            rendering_objects["xy_state_true_state_arrow"] = self.axes_dict[ax_name].arrow([], [], [], [], color="red", width=0.15*rendered_object_base_size*1.0)
            rendering_objects["xy_state_mean_predicted_state_arrow"] = self.axes_dict[ax_name].arrow([], [], [], [], color="tab:olive", width=0.15*rendered_object_base_size*1.0)
            rendering_objects["xy_state_true_state_circle"] = plt.Circle((0, 0), 0.25, color="red")
            rendering_objects["xy_state_mean_predicted_state_circle"] = plt.Circle((0, 0), 0.25, color="tab:olive")
            self.axes_dict[ax_name].add_patch(rendering_objects["xy_state_true_state_circle"])
            self.axes_dict[ax_name].add_patch(rendering_objects["xy_state_mean_predicted_state_circle"])

            # The particles rendering
            x = particles[0, :, 0].cpu().numpy()
            y = particles[0, :, 1].cpu().numpy()
            rendering_objects["xy_state_particles"] = self.axes_dict[ax_name].scatter(x, y, color="black", s=1)


            # All the objects that get updated on each frame
            rendering_objects_that_get_updated.append(rendering_objects["xy_state_true_state_arrow"])
            rendering_objects_that_get_updated.append(rendering_objects["xy_state_mean_predicted_state_arrow"])
            rendering_objects_that_get_updated.append(rendering_objects["xy_state_true_state_circle"])
            rendering_objects_that_get_updated.append(rendering_objects["xy_state_mean_predicted_state_circle"])
            rendering_objects_that_get_updated.append(rendering_objects["xy_state_particles"])


            # if(is_zoomed == False):


            #     # rendering_objects["xy_state_step_number_text"] = self.axes_dict[ax_name].text(1.5, 0.95, "", horizontalalignment='center', verticalalignment='center',transform=self.axes_dict[ax_name].transAxes, fontsize=15, fontweight="bold")
            #     # rendering_objects["xy_state_step_number_text"] = plt.figtext(0.5, 0.9, 'Figure text')
            #     # plt.figtext(0.5, 0.5, 'Figure text')

            #     text_str = "Step #{:02d}".format(0)
            #     rendering_objects["xy_state_step_number_text"] = plt.text(0.0, -2.0, text_str, horizontalalignment='center', fontsize=15, fontweight="bold")
            #     # rendering_objects_that_get_updated.append(rendering_objects["xy_state_step_number_text"])

            # Set some things to be static so that they dont constantly get updated
    
            # Set the title
            self.axes_dict[ax_name].set_title(title, fontweight="bold", fontsize=15)

            # Set the axis labels
            self.axes_dict[ax_name].set_xlabel("X", fontweight="bold", fontsize=15)
            self.axes_dict[ax_name].set_ylabel("Y", fontweight="bold", fontsize=15)

            if(is_zoomed == False):
                # Set the X and Y limits
                # self.axes_dict[ax_name].set_xlim([0, world_map_origional_size])
                # self.axes_dict[ax_name].set_ylim([0, world_map_origional_size])

                x_min, x_max, y_min, y_max = self.get_world_map_limits(world_map)

                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = int(max(x_range, y_range))
                max_range = max_range + 15
                max_range = min(max_range, world_height)
                max_range = min(max_range, world_width)

                x_center = (x_max + x_min) // 2
                y_center = (y_max + y_min) // 2

                x_min = x_center - (max_range // 2)
                x_min = max(x_min, 0)
                x_max = x_min + max_range
                x_max = min(x_max, world_width)
                x_min = x_max - max_range

                y_min = y_center - (max_range // 2)
                y_min = max(y_min, 0)
                y_max = y_min + max_range
                y_max = min(y_max, world_height)
                y_min = y_max - max_range

                self.axes_dict[ax_name].set_xlim([x_min, x_max])
                self.axes_dict[ax_name].set_ylim([y_min, y_max])


            return rendering_objects, rendering_objects_that_get_updated

        # Create the "posterior_xy" objects
        ro_dict, ro_update = create_posterior_xy_rendering_objects("posterior_xy", False, "Posterior XY")
        self.all_rendering_objects["posterior_xy"] = ro_dict
        self.all_rendering_objects_that_get_updated.extend(ro_update)

        # # Create the "zoomed_posterior_xy" objects
        ro_dict, ro_update = create_posterior_xy_rendering_objects("zoomed_posterior_xy", True, "Posterior XY (Zoomed)")
        self.all_rendering_objects["zoomed_posterior_xy"] = ro_dict
        self.all_rendering_objects_that_get_updated.extend(ro_update)


        ########################################################################################################################################################################
        ########################################################################################################################################################################
        # "posterior_angle" Axis Stuff
        ########################################################################################################################################################################
        ########################################################################################################################################################################
        rendering_objects = dict()

        rendering_objects["pdf"], = self.axes_dict["posterior_angle"].plot([], [], label="Angle PDF Value")
        self.all_rendering_objects_that_get_updated.append(rendering_objects["pdf"])

        thetas = particles[0, :, 2].cpu().numpy()
        rendering_objects["particle_vlines"] = self.axes_dict["posterior_angle"].vlines(thetas, 0, 1.0, color="black", linewidth=1, label="Particles")
        self.all_rendering_objects_that_get_updated.append(rendering_objects["particle_vlines"])

        thetas = particles[0, 0, 2].cpu().numpy()
        rendering_objects["mean_particle_vlines"] = self.axes_dict["posterior_angle"].vlines(thetas, 0, 1.0, color="tab:olive", linewidth=2, label="Mean Particle")
        self.all_rendering_objects_that_get_updated.append(rendering_objects["mean_particle_vlines"])

        thetas = particles[0, 0, 2].cpu().numpy()
        rendering_objects["true_state"] = self.axes_dict["posterior_angle"].vlines(thetas, 0, 1.0, color="red", linewidth=2, label="True Theta")
        self.all_rendering_objects_that_get_updated.append(rendering_objects["true_state"])


        self.all_rendering_objects["posterior_angle"] = rendering_objects

        # Set the things that should not change
        self.axes_dict["posterior_angle"].set_xlim([-np.pi, np.pi])
        leg = self.axes_dict["posterior_angle"].legend(loc="upper left", prop={'size': 10, "weight":"bold"})
        self.axes_dict["posterior_angle"].set_title("Posterior Angle", fontsize=15, fontweight="bold")
        self.axes_dict["posterior_angle"].set_xlabel("θ", fontsize=15, fontweight="bold")
        self.axes_dict["posterior_angle"].set_ylabel("Probability Density Function Value", fontsize=10, fontweight="bold")

        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)




        ########################################################################################################################################################################
        ########################################################################################################################################################################
        # "observation" Axis Stuff
        ########################################################################################################################################################################
        ########################################################################################################################################################################
        rendering_objects = dict()
        rendering_objects["observation"] = self.axes_dict["observation"].imshow(np.zeros((2, 2)))
        self.all_rendering_objects_that_get_updated.append(rendering_objects["observation"])
        self.all_rendering_objects["observation"] = rendering_objects

        # Set the properties of the rendering
        # self.axes_dict["observation"].axis('off')
        self.axes_dict["observation"].set_title("Current Camera\nObservation", fontsize=15, fontweight="bold")

        self.axes_dict["observation"].tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False) 


        self.fig.tight_layout()

        return self.all_rendering_objects_that_get_updated

    def render_frame(self, frame_number, data, output_dicts, axes):

        # Extract stuff if we have them
        particle_weights = self.extract_if_present("particle_weights", output_dicts)


        # # Add a text label indicating what frame number we are at
        # text_str = "Step #{:02d}".format(frame_number)
        # self.all_rendering_objects["posterior_xy"]["xy_state_step_number_text"].set_text(text_str)

        # Render the xy state
        self.draw_xy_state(frame_number, self.axes_dict["posterior_xy"], "posterior_xy", data, output_dicts)

        # Render the zoomed xy state
        self.draw_xy_state(frame_number, self.axes_dict["zoomed_posterior_xy"], "zoomed_posterior_xy", data, output_dicts, do_zoom=True)

        # Render the angle
        self.draw_angle_state(frame_number, self.axes_dict["posterior_angle"], "posterior_angle", data, output_dicts, "Posterior Angle")

        # Draw the observation over time
        self.draw_observation(self.axes_dict["observation"], data, frame_number)

        self.fig.tight_layout()

        return self.all_rendering_objects_that_get_updated

    def draw_xy_state(self, frame_number, ax, ax_name,  data, output_dicts, key_name_prefix="", do_zoom=False):

        # Extract the data
        world_map = data["world_map"]
        states = data["states"]

        # Get the current true state
        current_true_state = states[0, frame_number]

        # Get the width and the height of the world map
        world_width = world_map.shape[2]
        world_height = world_map.shape[1]

        # Compute the render-able area
        if(do_zoom):
            
            # How much to zoom
            ZOOM_FACTOR = 10

            # Compute the new widths
            new_world_width = world_width / ZOOM_FACTOR
            new_world_height = world_height / ZOOM_FACTOR

            # Get the current position of the state so we can center it
            x_state = current_true_state[0].item()
            y_state = current_true_state[1].item()

            # Compute the new mins and maxs
            x_min = x_state - (new_world_width/2)
            x_max = x_state + (new_world_width/2)
            y_min = y_state - (new_world_height/2)
            y_max = y_state + (new_world_height/2)

            # Make sure they are in range
            if(x_min < 0):
                x_max = x_max - x_min
                x_min = x_min - x_min

            if(y_min < 0):
                y_max = y_max - y_min
                y_min = y_min - y_min

            if(x_max >= world_width):
                diff = x_max - world_width
                x_max -= diff
                x_min -= diff

            if(y_max >= world_height):
                diff = y_max - world_height
                y_max -= diff
                y_min -= diff


        else:
            x_min = 0
            x_max = world_width

            y_min = 0
            y_max = world_height

        # Compute the base size that we will use scaling all the rendered objects
        max_range = max((x_max-x_min), (y_max-y_min))

        if(do_zoom):
            rendered_object_base_size = max_range / 20.0
        else:
            rendered_object_base_size = max_range / 40.0

        # Unpack the data
        particles = output_dicts["particles"]
        particle_weights = self.get_particle_weights(output_dicts, key_name_prefix)

        # Get the data for this frame only
        particles = particles[frame_number]
        particle_weights = particle_weights[frame_number]

        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Get the bandwidths since we have one (or one is manually set) and get the one for this frame only
            bandwidths = self.get_bandwidths(output_dicts, key_name_prefix)
            bandwidths = bandwidths[frame_number]

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            samples = kde.sample((100000, ))

            # Draw the probability distribution of where we think we should put mass
            x_samples = samples[0, :, 0].cpu().numpy()
            y_samples = samples[0, :, 1].cpu().numpy()
            ranges = [[0, world_width], [0, world_height]]
            H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=1000, range=ranges)
            self.all_rendering_objects[ax_name]["xy_state_kde_histogram"].set_data(H.T)
            self.all_rendering_objects[ax_name]["xy_state_kde_histogram"].autoscale()

        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Draw the particles
            # x = particles[0, :, 0].cpu().numpy()
            # y = particles[0, :, 1].cpu().numpy()
            offsets = particles[0, :, 0:2].cpu().numpy()
            self.all_rendering_objects[ax_name]["xy_state_particles"].set_offsets(offsets)

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(particles)
            predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=1)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

            # Render the mean particle
            arrow_obj = self.all_rendering_objects[ax_name]["xy_state_mean_predicted_state_arrow"]
            circle_obj = self.all_rendering_objects[ax_name]["xy_state_mean_predicted_state_circle"]
            self.draw_state_with_arrow(ax, arrow_obj, circle_obj, predicted_state[0], rendered_object_base_size)

        # # Plot the true location    
        arrow_obj = self.all_rendering_objects[ax_name]["xy_state_true_state_arrow"]
        circle_obj = self.all_rendering_objects[ax_name]["xy_state_true_state_circle"]
        self.draw_state_with_arrow(ax, arrow_obj, circle_obj, current_true_state, rendered_object_base_size)

        # Set the X and Y limits if we are zoomed in
        if(do_zoom):
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

    def draw_angle_state(self, frame_number, ax, ax_name, data, output_dicts, title, key_name_prefix=""):

        # Unpack the data
        particles = output_dicts["particles"]
        particle_weights = self.get_particle_weights(output_dicts, key_name_prefix)

        # Get the data for this frame only
        particles = particles[frame_number]
        particle_weights = particle_weights[frame_number]

        # Figure out the max height so we can scale things
        max_height = 1.0

        # If we have the KDE then do it
        if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

            # Get the bandwidths since we have one (or one is manually set) and get the one for this frame only
            bandwidths = self.get_bandwidths(output_dicts, key_name_prefix)
            bandwidths = bandwidths[frame_number]

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

            # compute the log prob for samples over the unit circle
            x = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(0).unsqueeze(-1).to(self.device)
            log_probs = kde.marginal_log_prob(x, [0,1])

            # Convert log_probs from log space to normal space
            probs = torch.exp(log_probs)

            # Plot the probabilities
            self.all_rendering_objects[ax_name]["pdf"].set_data(x.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy())

            # update the max height
            max_height = torch.max(probs).item()

        # Scale to the max height
        ax.set_ylim([0, max_height])

        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):
            # Draw the particles
            thetas = particles[0, :, 2].cpu().numpy()
            # ax.vlines(thetas, 0, 0.25*max_height, color="black", linewidth=1, label="Particles")
            seg_new = [np.array([[xx, 0],[xx, max_height*0.25]]) for xx in thetas]
            self.all_rendering_objects[ax_name]["particle_vlines"].set_segments(seg_new)

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(particles)
            predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=1)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

            # Draw the mean particle
            theta = predicted_state[0, 2].cpu().numpy()
            # ax.vlines(theta, 0, 1.0*max_height, color="tab:olive", linewidth=2, label="Mean Particle")
            seg_new = [np.array([[theta, 0],[theta, max_height*1.0]])]
            self.all_rendering_objects[ax_name]["mean_particle_vlines"].set_segments(seg_new)

        # Draw a vertical line where the true state is
        states = data["states"]
        true_theta = states[0, frame_number,2].item()
        seg_new = [np.array([[true_theta, 0],[true_theta, max_height*1.0]])]
        self.all_rendering_objects[ax_name]["true_state"].set_segments(seg_new)


    def draw_full_xy_path(self, ax, data, output_dicts, key_name_prefix="", linewidth=2.0):

        # Extract the data
        states = data["states"]

        # Unpack the data
        particles = output_dicts["particles"]
        particle_weights = self.get_particle_weights(output_dicts, key_name_prefix)
     
         # Draw the true path
        x1 = states[0, : ,0].numpy()
        y1 = states[0, : ,1].numpy()
        ax.plot(x1, y1, color="red", label="True State", alpha=1.0, linewidth=linewidth)

        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(particles)
            predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=2)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

            # Draw the Predicted path
            x1 = predicted_state[:,0,0].detach().cpu().numpy()
            y1 = predicted_state[:,0,1].detach().cpu().numpy()
            ax.plot(x1, y1, color="tab:olive", label="Mean Particle", alpha=1.0, linewidth=linewidth)


        # Add the legend
        # tmp = self.legend_without_duplicate_labels(ax)
        # tmp = (*tmp,)
        # tmp = (*zip(tmp),)
        # ax.legend(tmp, loc="lower right", prop={'size': 9})
        leg = ax.legend(loc="lower right", prop={'size': 10, "weight":"bold"},)
        
        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)


    def draw_observation(self, ax, data, frame_number):

        # Extract the data
        observations = data["observations"]

        # Remove the batch dim 
        observations = observations[0]

        # Get the observation for this timestep
        observation = observations[frame_number]

        # Render the observation
        self.all_rendering_objects["observation"]["observation"].set_data(observation.permute((1, 2, 0)).numpy())
        self.all_rendering_objects["observation"]["observation"].autoscale()

        text_str = "Time-step {:02d}".format(frame_number)
        ax.set_xlabel(text_str, fontweight="bold", fontsize=15)



    def get_particle_weights(self, output_dicts, key_name_prefix=""):

        # Extract the particle weights with the prefix, otherwise just use the other weights
        if("{}particle_weights".format(key_name_prefix) in output_dicts):
            particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
        else:
            particle_weights = output_dicts["particle_weights"] 

        return particle_weights

    def get_bandwidths(self, output_dicts, key_name_prefix=""):

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

        return bandwidths

    def draw_state_with_arrow(self, ax, arrow_obj, circle_obj, state, rendered_object_base_size):
        x = state[0].item()
        y = state[1].item()
        circle_obj.center = (x,y)
        # circle = plt.Circle((x, y), 0.25*size_scaler, color=color)
        # ax.add_patch(circle)
        dy = torch.sin(state[2]).item() * 1.0 * rendered_object_base_size
        dx = torch.cos(state[2]).item() * 1.0 * rendered_object_base_size
        arrow_obj.set_data(x=x, y=y, dx=dx, dy=dy)

    def legend_without_duplicate_labels(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        return unique

    def render_world_map(self, ax, world_map):

        # Render the world map, need to theshold to make sure the walls come up nice an sharp
        world_map_render = world_map.squeeze(0).numpy()
        mask1 = world_map_render>0.0001
        mask2 = world_map_render<0.0001
        world_map_render[mask1] = 1.0
        world_map_render[mask2] = 0.0

        # Fix the borders of the world map
        world_map_render = self.make_world_map_borders_white(world_map_render)

        # Stack into RGB alpha?
        world_map_render = np.dstack((world_map_render, world_map_render, world_map_render, world_map_render<0.5))

        # Get the width and the height of the world map
        world_width = world_map.shape[2]
        world_height = world_map.shape[1]

        # Draw the world map
        return ax.imshow(world_map_render, cmap="gray", extent=[0, world_width, 0, world_height], origin = "lower", aspect="auto")


    def get_world_map_limits(self, world_map):

        if(torch.is_tensor(world_map)):
            world_map = world_map.squeeze(0).numpy()
        mask1 = world_map>0.0001
        mask2 = world_map<0.0001
        world_map[mask1] = 1.0
        world_map[mask2] = 0.0

        H, W = world_map.shape


        ####################################################################
        ## Bottom Up
        ####################################################################
        for y in range(H):

            row = world_map[y, :]
            if(np.sum(row) != 0):
                break

        y_min = y

        ####################################################################
        ## Top Down
        ####################################################################
        for y in range(H-1, 1, -1):

            row = world_map[y, :]
            if(np.sum(row) != 0):
                break

        y_max = y

        ####################################################################
        ## Left to Right
        ####################################################################
        for x in range(W):

            col = world_map[:, x]
            if(np.sum(col) != 0):
                break

        x_min = x
        
        ####################################################################
        ## Right to Left
        ####################################################################
        for x in range(W-1, 1, -1):

            col = world_map[:, x]
            if(np.sum(col) != 0):
                break

        x_max = x

        return x_min, x_max, y_min, y_max


    def make_world_map_borders_white(self, world_map_render):

        x_min, x_max, y_min, y_max = self.get_world_map_limits(world_map_render)

        world_map_render_new = np.copy(world_map_render)

        edge_border_width = 15

        H, W = world_map_render.shape

        # White it out
        world_map_render_new[0:y_min, :] = 1.0
        world_map_render_new[y_max:H, :] = 1.0
        world_map_render_new[:, 0:x_min] = 1.0
        world_map_render_new[:, x_max:W] = 1.0

        # Make the Border
        world_map_render_new[max(y_min-edge_border_width, 0):y_min, x_min:x_max] = 0.0
        world_map_render_new[y_max:min(y_max+edge_border_width, H), x_min:x_max] = 0.0
        world_map_render_new[y_min:y_max, max(x_min-edge_border_width, 0):x_min] = 0.0
        world_map_render_new[y_min:y_max, x_max:min(x_max+edge_border_width, W)] = 0.0


        return world_map_render_new



















    def render_panel(self, render_index, data, output_dicts):

        # Get the sequence index, if one doesnt exist then just use the render index
        sequence_index = render_index
        if("dataset_index" in data):
            sequence_index = data["dataset_index"][0]

        # Compute the scaling factors for the figure
        world_map = data["world_map"][0]
        x_lim, y_lim = self.get_map_render_limits(world_map.numpy())
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        x_to_y_ratio = float(y_range) / float(x_range)



        # Make the figure
        rows = self.render_panel_num_rows
        cols = self.render_panel_num_cols
        number_to_render = min((rows * cols), data["states"].shape[1])
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols, (3*rows*x_to_y_ratio) + 0.5), squeeze=False)
        # fig, axes = plt.subplots(rows, cols, sharex=False, squeeze=False)
        axes = axes.reshape(-1,)

        # Compute the Render Indices
        if(self.render_panel_modulo == 1):
            indices_to_render = [i for i in range(1, number_to_render+1)]
        else:
            indices_to_render = [i*self.render_panel_modulo for i in range(0, number_to_render-len(self.render_panel_must_include_indices)+1)]
            indices_to_render.extend(self.render_panel_must_include_indices)
            indices_to_render[0] += 1
            indices_to_render.sort()


        # Draw all the timesteps
        frame_number = 0 
        max_rendered_frame_number = 0
        for i in tqdm(range(number_to_render), leave=False):
            frame_number = indices_to_render[i]
            max_rendered_frame_number = max(frame_number, max_rendered_frame_number)

            ax = axes[i]
            true_state_arrow, mean_particle_arrow, particles_scatter = self.draw_xy_state_for_pannel_rendering(frame_number, ax, data, output_dicts)

            # Plot the observation as an inset
            # inset_ax = ax.inset_axes([0.58,0.03,0.5,0.5])
            inset_ax = ax.inset_axes([0.7,0.01,0.3,0.3])
            observation = torch.permute(data["observations"][0, frame_number].cpu(), (1, 2, 0))
            # observation[observation<0] = 0
            # observation[observation>1.0] = 1.0
            inset_ax.imshow(observation.numpy())
            inset_ax.set_yticklabels([])
            inset_ax.set_xticklabels([])
            inset_ax.tick_params(left=False, bottom=False)
            inset_ax.patch.set_edgecolor('black')  
            inset_ax.patch.set_linewidth(3)  





        # Add Legend
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height, width=height*0.25)
            return p
            
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
            return p



        handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="black")]
        labels = ["True State", "Mean Particle", "Particles"]
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
        lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=14)
        fig.suptitle(' ', fontsize=15)

        # Adjust whitespace
        fig.tight_layout(rect=(0,0,1.0,1.0))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)

        # # Save the figure as a pdf
        save_file = "{}/panel_rendering_{}.pdf".format(self.save_dir, render_index)
        fig.savefig(save_file, format="pdf")


        # plt.show()


    def draw_xy_state_for_pannel_rendering(self, frame_number, ax, data, output_dicts, key_name_prefix="", do_zoom=True, render_paths=True):

        # Extract the data
        world_map = data["world_map"]
        states = data["states"]
        world_map_origional_size = data["world_map_origional_size"]

        # Remove the batch dim
        world_map = world_map.squeeze(0)
        states = states.squeeze(0)
        world_map_origional_size = world_map_origional_size.squeeze(0)

        # Convert the world map to numpy so we can fully process it and render it
        world_map = world_map.numpy()

        # Threshold the world map to make it sharper
        world_map[world_map > 0.1] = 1.0

        world_width = world_map.shape[1]
        world_height = world_map.shape[0]

        # Compute the render-able area
        if(do_zoom):
            
            # Get the render limits for the world map
            x_lim, y_lim = self.get_map_render_limits(world_map)
            # y_lim, x_lim = self.get_map_render_limits(world_map)

            # Get the range
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            
            # How much to zoom
            ZOOM_FACTOR = 5.0

            # Compute the new widths
            new_x_range = x_range / ZOOM_FACTOR
            new_y_range = y_range / ZOOM_FACTOR

            # Get the current position of the state so we can center it
            current_true_state = states[frame_number]
            x_state = current_true_state[0].item()
            y_state = current_true_state[1].item()

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(output_dicts["particles"][frame_number][0])
            predicted_state = torch.sum(particles * output_dicts["particle_weights"][frame_number][0].unsqueeze(-1), dim=0)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state).cpu()

            mean_state = (current_true_state + predicted_state) / 2.0
            x_state = mean_state[0].item()
            y_state = mean_state[1].item()


            # Compute the new mins and maxs
            x_min = x_state - (new_x_range/2)
            x_max = x_state + (new_x_range/2)
            y_min = y_state - (new_y_range/2)
            y_max = y_state + (new_y_range/2)

            # Make sure they are in range
            if(x_min < x_lim[0]):
                d = x_lim[0] - x_min
                x_max = x_max + d
                x_min = x_min + d

            if(y_min < y_lim[0]):
                d = y_lim[0] - y_min
                y_max = y_max + d
                y_min = y_min + d

            if(x_max >= x_lim[1]):
                diff = x_max - x_lim[1]
                x_max -= diff
                x_min -= diff

            if(y_max >= y_lim[1]):
                diff = y_max - y_lim[1]
                y_max -= diff
                y_min -= diff

            x_lim = [int(x_min), int(x_max)]
            y_lim = [int(y_min), int(y_max)]


        else:

            # Get the render limits for the world map
            x_lim, y_lim = self.get_map_render_limits(world_map)





        # Convert 
        world_map_rgb = np.zeros((world_map.shape[0], world_map.shape[1], 3))
        world_map_rgb[world_map>0, :] = 255

        # Get the range
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]

        # Render the world map
        # ax.imshow(world_map, cmap="gray")
        ax.imshow(world_map_rgb.astype("int"), origin="lower")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(3)  

        # Set the render limits for this plot
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Set the render base size
        max_range = max(x_range, y_range)
        rendered_object_base_size = max_range / 20.0



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
                    bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1]), device=particles.device)
                    for i in range(len(self.manual_bandwidth)):
                        bandwidths[...,i] = self.manual_bandwidth[i]

                elif(self.manual_bandwidth is not None):
                    bandwidths = torch.full(size=(particles.shape[0], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)


            else:
                print("Need a bandwidth from somewhere?!?!?!")
                assert(False)


            # Extract data we need for the frame number we need
            # particles = particles[frame_number]
            # particle_weights = particle_weights[frame_number]
            # bandwidths = bandwidths[frame_number]



            # Create the sampling mesh grid
            density = 100
            x = torch.linspace(x_lim[0], x_lim[1], x_range)
            y = torch.linspace(y_lim[0], y_lim[1], y_range)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

            # Create the KDE and compute the probs
            kde_params_copy = copy.deepcopy(self.kde_params)
            del kde_params_copy["dims"][2]
            kde = KernelDensityEstimator(kde_params_copy, particles[frame_number,:, :,0:2], particle_weights[frame_number], bandwidths[frame_number,:, 0:2])
            log_probs = kde.log_prob(points.unsqueeze(0).to(particles.device)).squeeze(0)
            probs = torch.exp(log_probs).cpu()

            # Scale
            probs_min = torch.min(probs)
            probs_max = torch.max(probs)
            probs -= probs_min
            probs /= (probs_max - probs_min)

            for i in range(points.shape[0]):
                p = points[i].numpy()
                c = probs[i].item()
                # world_map_rgb[int(p[1]), int(p[0]), 0:2] = 255.0 - (c*255.0)
                world_map_rgb[int(p[1]), int(p[0]), 0:2] = 255.0 - (c*255.0)

            world_map_rgb[world_map==0, :] = 0
            ax.imshow(world_map_rgb.astype("int"), origin="lower")

        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            if(render_paths):
                # Compute the mean particle
                particles = output_dicts["particles"]
                particle_weights = output_dicts["particle_weights"]
                predicted_state = torch.sum(particles[...,0:2] * particle_weights.unsqueeze(-1), dim=2)

                # Draw the Predicted path
                x1 = predicted_state[:, 0,0].detach().cpu().numpy()
                y1 = predicted_state[:, 0,1].detach().cpu().numpy()
                ax.plot(x1, y1, color=sns.color_palette("bright")[8], label="Mean Particle", alpha=1.0, linewidth=0.75)


            # Get the right set of particles.  
            particles = particles[frame_number][0]
            particle_weights = particle_weights[frame_number][0]

            # Draw the particles
            x = particles[:, 0].cpu().numpy()
            y = particles[:, 1].cpu().numpy()
            particles_scatter = ax.scatter(x,y, color="black", s=0.3*rendered_object_base_size)

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(particles)
            predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=0)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

            # Render the mean particle
            x = predicted_state[0].item()
            y = predicted_state[1].item()
            mean_particle_circle = plt.Circle((x, y), 0.25*rendered_object_base_size, color=sns.color_palette("bright")[8])
            ax.add_patch(mean_particle_circle)
            dy = torch.sin(predicted_state[2]).item() * 1.0*rendered_object_base_size
            dx = torch.cos(predicted_state[2]).item() * 1.0*rendered_object_base_size
            mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.25*rendered_object_base_size)

        if(render_paths):
            # Draw the true path
            x1 = states[: ,0].numpy()
            y1 = states[: ,1].numpy()
            ax.plot(x1, y1, color="red", label="True State", alpha=1.0, linewidth=0.75)


        # Plot the true location
        true_x = states[frame_number,0].item()
        true_y = states[frame_number,1].item()
        true_state_circle = plt.Circle((true_x, true_y), 0.25 * rendered_object_base_size, color=sns.color_palette("bright")[3])
        ax.add_patch(true_state_circle)
        dy = torch.sin(states[frame_number,2]).item() * 1.0 * rendered_object_base_size
        dx = torch.cos(states[frame_number,2]).item() * 1.0 * rendered_object_base_size
        true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.25*rendered_object_base_size)

        
        # Add a text label indicating what frame number we are at
        text_str = "Time-step #{:02d}".format(frame_number)
        ax.text(0.05, 0.065, text_str, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, weight="bold", fontsize=12, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
        # ax.set_xlabel(text_str, weight="bold", fontsize=12)

        return true_state_arrow, mean_particle_arrow, particles_scatter


    def get_map_render_limits(self, world_map):
        ''' Taken from:
                https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
        '''

        # Find all non-zero points (text)
        coords = cv2.findNonZero(world_map) 

        # Find minimum spanning bounding box
        x, y, w, h = cv2.boundingRect(coords) 

        # Add some padding
        padding = 0
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = w + padding
        h = h + padding

        x_lim = (x, x+w)
        y_lim = (y, y+h)

        return x_lim, y_lim


