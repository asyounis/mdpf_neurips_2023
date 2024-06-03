# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_video_base import *
from models.sequential_models import *

class BearingsOnlyFullSequenceEvaluation(FullSequenceEvaluationVideoBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)


        self.axes_dict = None
        self.fig = None


    def new_rendering(self):
        self.axes_dict = None

    def get_rows_and_cols(self):
            
        # 1 - State 
        # 2 - State angle
        # 3 - ESS
        rows = 3
        cols = 2

        # if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):
            # cols += 1

        return rows, cols


    def extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None

    def render_frame(self, frame_number, data, output_dicts, axes):

        if(frame_number == 0):
            self.init_frame(axes)

        # Render the xy state
        self.draw_xy_state(frame_number, self.axes_dict["posterior_xy"], data, output_dicts, "Posterior XY")

        # Render the angle
        self.draw_angle_state(frame_number, self.axes_dict["posterior_angle"], data, output_dicts, "Posterior Angle")
        

    def init_frame(self, axes):


        if(self.axes_dict is not None):
            return 

        # For this one, reshape back into rows and cols
        rows, cols = self.get_rows_and_cols()
        axes = axes.reshape((rows, cols))

        # get the figure
        fig = axes[0, 0].get_figure()

        # Helper funcs
        def remove_axis(ax_to_remove):
            ax_flat = ax_to_remove.reshape(-1,)
            for ax in ax_flat:
                ax.remove()


        # Get the gridspec of the axis so we can use that to create larger subplots 
        gridspec = axes[0, 0].get_gridspec()

        # Put the axis into the dict
        self.axes_dict = dict()


        # Posterior Plots
        #################################################
        remove_axis(axes[0:2, 0:2])
        self.axes_dict["posterior_xy"] = fig.add_subplot(gridspec[0:2, 0:2])

        remove_axis(axes[2:3, 0:2])
        self.axes_dict["posterior_angle"] = fig.add_subplot(gridspec[2:3, 0:2])

        # Change the figure size
        fig.set_size_inches(8, 9)



    def render_panel(self, render_index, data, output_dicts):

        # Get the sequence index, if one doesnt exist then just use the render index
        sequence_index = render_index
        if("dataset_index" in data):
            sequence_index = data["dataset_index"][0]

        # Make the figure
        rows = self.render_panel_num_rows
        cols = self.render_panel_num_cols
        number_to_render = (rows * cols)
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols*0.93, 3*rows))
        axes = axes.reshape(-1,)


        
        # Compute the Render Indices
        indices_to_render = [i*self.render_panel_modulo for i in range(0, number_to_render-len(self.render_panel_must_include_indices)+1)]
        indices_to_render.extend(self.render_panel_must_include_indices)
        indices_to_render[0] += 1
        indices_to_render.sort()



        # Draw all the timesteps
        frame_number = 0 
        max_rendered_frame_number = 0
        for i in tqdm(range(number_to_render), leave=False, desc="Panel Rendering"):
            frame_number = indices_to_render[i]
            max_rendered_frame_number = max(frame_number, max_rendered_frame_number)

            ax = axes[i]
            true_state_arrow, mean_particle_arrow, particles_scatter = self.draw_xy_state(frame_number, ax, data, output_dicts, is_panel=True)

        # Format the axis
        for ax in axes:

            # Turn off the axis labels
            # ax.axis('off')

            # Set the aspect ratio to a square 
            ax.set_aspect("equal")
            
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            # turn off the axis ticks little bars
            ax.tick_params(left=False, bottom=False)

            # Add a box around the outside
            ax.patch.set_edgecolor('black')  
            ax.patch.set_linewidth(1)  


        # Add Legend
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height, width=height*0.25)
            return p
            
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
            return p



        handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="black"), plt.Circle((0, 0), 0.25, color="green")]
        labels = ["True State", "Mean Particle", "Particles", "Radar Station + Radar Sensor Reading"]
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
        lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=14)

        # Adjust whitespace
        fig.subplots_adjust(wspace=0.01, hspace=0.03)
        fig.tight_layout(rect=(0,0,1,0.945))

        # Save the figure as a png
        save_file = "{}/panel_rendering_{}.png".format(self.save_dir, render_index)
        fig.savefig(save_file)

        # Save the figure as a pdf
        save_file = "{}/panel_rendering_{}.pdf".format(self.save_dir, render_index)
        fig.savefig(save_file, format="pdf")




    def draw_xy_state(self, frame_number, ax, data, output_dicts, title=None, key_name_prefix="", is_panel=False):

        if(is_panel):
            rendered_object_base_size = 2.5
        else:
            # rendered_object_base_size = 1.0
            rendered_object_base_size = 1.5

        x_lim = self.evaluation_dataset.get_x_range()
        y_lim = self.evaluation_dataset.get_y_range()

        # Set the X and Y limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


        # Extract the data we need for rendering
        observations = data["observations"]
        states = data["states"]

        # Transform the observations
        transformed_observation = self.problem.observation_transformer.forward_tranform(observations)

        # Draw the sensor information 
        sensors = self.evaluation_dataset.get_sensors()
        for i, sensor in enumerate(sensors):

            # The sensor location
            position = sensor.get_position()
            x = position[0].item()
            y = position[1].item()
            ax.add_patch(plt.Circle((x, y), 0.5*rendered_object_base_size, color='tab:green'))

            # The sensor observation
            sensor_obs_y = transformed_observation[0, frame_number, i].item() * 20
            sensor_obs_x = transformed_observation[0, frame_number, int(transformed_observation.shape[-1] / 2) + i].item() * 20
            x_values = [x, sensor_obs_x]
            y_values = [y, sensor_obs_y]
            ax.plot(x_values, y_values, color="tab:green", linewidth=rendered_object_base_size*2.0, label="Observation")





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


            # Create the sampling mesh grid
            density = 400
            x = torch.linspace(x_lim[0], x_lim[1], density)
            y = torch.linspace(y_lim[0], y_lim[1], density)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

            # Create the KDE and compute the probs
            kde_params_copy = copy.deepcopy(self.kde_params)
            del kde_params_copy["dims"][2]
            kde = KernelDensityEstimator(kde_params_copy, particles[:, :,0:2], particle_weights, bandwidths[:, 0:2])
            log_probs = kde.log_prob(points.unsqueeze(0).to(particles.device)).squeeze(0)
            probs = torch.exp(log_probs).cpu()

            # Scale
            probs_min = torch.min(probs)
            probs_max = torch.max(probs)
            probs -= probs_min
            probs /= (probs_max - probs_min)
            probs = probs.reshape(grid_x.shape)


            probs = np.transpose(probs, axes=[1,0])
            ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True))


        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):


            # Extract the data that we need
            particles = output_dicts["particles"]
            particles = particles[frame_number][0]

            # Extract the particle weights with the prefix, otherwise just use the other weights
            if("{}particle_weights".format(key_name_prefix) in output_dicts):
                particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
            else:
                particle_weights = output_dicts["particle_weights"] 
            particle_weights = particle_weights[frame_number]

            # Draw the particles
            x = particles[:, 0].cpu().numpy()
            y = particles[:, 1].cpu().numpy()
            particles_scatter = ax.scatter(x,y, color="black", s=1.0*rendered_object_base_size, label="Particles")

            # Compute the mean particle
            particles = self.model.particle_transformer.backward_tranform(particles)
            predicted_state = torch.sum(particles * particle_weights[0].unsqueeze(-1), dim=0)
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

            # Render the mean particle
            x = predicted_state[0].item()
            y = predicted_state[1].item()
            mean_particle_circle = plt.Circle((x, y), 0.25*rendered_object_base_size, color=sns.color_palette("bright")[8])
            ax.add_patch(mean_particle_circle)
            dy = torch.sin(predicted_state[2]).item() * 1.0*rendered_object_base_size
            dx = torch.cos(predicted_state[2]).item() * 1.0*rendered_object_base_size
            mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.15*rendered_object_base_size)


        if(self.model.outputs_single_solution()):

            # Extract the predicted state
            predicted_state = output_dicts["predicted_state"]

            # Plot the true location
            x = predicted_state[frame_number,0,0].item()
            y = predicted_state[frame_number,0,1].item()
            true_state_circle = plt.Circle((x, y), 0.25 * rendered_object_base_size, color='blue')
            ax.add_patch(true_state_circle)
            dy = torch.sin(predicted_state[frame_number,0,2]).item() * 1.0 * rendered_object_base_size
            dx = torch.cos(predicted_state[frame_number,0,2]).item() * 1.0 * rendered_object_base_size
            ax.arrow(x, y, dx, dy, color='blue', width=0.15*rendered_object_base_size)




        # Plot the true location
        true_x = states[0,frame_number,0].item()
        true_y = states[0,frame_number,1].item()
        true_state_circle = plt.Circle((true_x, true_y), 0.25 * rendered_object_base_size, color=sns.color_palette("bright")[3])
        ax.add_patch(true_state_circle)
        dy = torch.sin(states[0,frame_number,2]).item() * 1.0 * rendered_object_base_size
        dx = torch.cos(states[0,frame_number,2]).item() * 1.0 * rendered_object_base_size
        true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.15*rendered_object_base_size)

        if(is_panel):
            # Add a text label indicating what frame number we are at
            text_str = "Time-step #{:02d}".format(frame_number)
            ax.text(0.5, 0.93, text_str, horizontalalignment="center", verticalalignment="center",transform=ax.transAxes, weight="bold", fontsize=12)

            return true_state_arrow, mean_particle_arrow, particles_scatter


        else:
            # Add a text label indicating what frame number we are at
            text_str = "Time-step {:03d}".format(frame_number)
            ax.text(0.15, 0.78, text_str, horizontalalignment="center", verticalalignment="center",transform=ax.transAxes, weight="bold", fontsize=12)
            
            ax.set_title(title, fontweight="bold", fontsize=15)
            ax.set_xlabel("X", fontweight="bold", fontsize=15)
            ax.set_ylabel("Y", fontweight="bold", fontsize=15)

            





            def make_legend_arrow(legend, orig_handle,xdescent, ydescent, width, height, fontsize):
                p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.95*height, width=3.0)
                return p




            handles, labels = ax.get_legend_handles_labels()

            handles.append(true_state_arrow)
            labels.append("True State")

            handles.append(mean_particle_arrow)
            labels.append("Mean Particle")


            handler_map = dict()
            handler_map[mpatches.FancyArrow] = HandlerPatch(patch_func=make_legend_arrow)

            legend_properties = {'weight':'bold'}
            leg = ax.legend(handles, labels, handler_map=handler_map, loc="upper left", fontsize=12, prop=legend_properties)

            # change the line width for the legend
            for line in leg.get_lines():
                line.set_linewidth(4.0)


            leg.legendHandles[1]._sizes = [45]


            # ax.legend(loc="upper right")



    def draw_angle_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

        # Set the X limits for the full unit circle
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([0, 1.0])


        # Figure out the max height so we can scale things
        max_height = 1.0

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

            # Use a KDE to generate samples that we will use in the histogram
            kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

            # compute the log prob for samples over the unit circle
            x = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(0).unsqueeze(-1).to(self.device)
            log_probs = kde.marginal_log_prob(x, [0,1])

            # Convert log_probs from log space to normal space
            probs = torch.exp(log_probs)

            # Plot the probabilities
            ax.plot(x.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy(), label="Angle PDF Value", linewidth=3.0)

            ax.set_ylim([0, torch.max(probs).item()])

            # update the max height
            max_height = torch.max(probs).item()


        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"]
            particles = particles[frame_number]

            # Draw the particles
            thetas = particles[0, :, 2].cpu().numpy()
            ax.vlines(thetas, 0, max_height*0.25, color="black", linewidth=2, label="Particles")

        # If we output a single solution
        if(self.model.outputs_single_solution()):

            # Extract the predicted state
            predicted_state = output_dicts["predicted_state"]

            # Plot the estimated location
            theta = predicted_state[frame_number,0,2].item()
            theta = np.asarray(theta)
            ax.vlines(theta, 0, max_height, color="blue", linewidth=3, label="Predicted Theta")


        # Draw a vertical line where the true state is
        states = data["states"]
        true_theta = states[0, frame_number,2].item()
        true_theta = np.asarray(true_theta)
        ax.vlines(true_theta, 0, max_height, color="red", linewidth=3, label="True Theta")


        ax.set_title(title, fontweight="bold", fontsize=15)
        ax.set_xlabel("θ", fontsize=15, fontweight="bold")
        ax.set_ylabel("PDF Value", fontsize=12, fontweight="bold")

        legend_properties = {'weight':'bold'}
        leg = ax.legend(loc="upper left", fontsize=12, prop=legend_properties)

        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)















    # def draw_xy_state(self, frame_number, ax, data, output_dicts, title=None, key_name_prefix="", is_panel=False):

    #     if(is_panel):
    #         rendered_object_base_size = 2.5
    #     else:
    #         rendered_object_base_size = 1.0

    #     x_lim = self.evaluation_dataset.get_x_range()
    #     y_lim = self.evaluation_dataset.get_y_range()

    #     # Set the X and Y limits
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)

    #     # If we have the KDE then do it
    #     if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

    #         # Extract the data that we need
    #         particles = output_dicts["particles"]

    #         # Extract the particle weights with the prefix, otherwise just use the other weights
    #         if("{}particle_weights".format(key_name_prefix) in output_dicts):
    #             particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
    #         else:
    #             particle_weights = output_dicts["particle_weights"] 

    #         # If we are outputting a KDE then we have a bandwidth
    #         if(self.model.outputs_kde()):
    #             # Extract the bandwidths with the prefix, otherwise just use the other bandwidth
    #             if("{}bandwidths".format(key_name_prefix) in output_dicts):
    #                 bandwidths = output_dicts["{}bandwidths".format(key_name_prefix)] 
    #             else:
    #                 bandwidths = output_dicts["bandwidths"] 

    #         elif(self.use_manual_bandwidth and (self.manual_bandwidth is not None)):

    #             # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
    #             if(isinstance(self.manual_bandwidth, list)):

    #                 # Make sure they specified the correct number of bandwidths
    #                 assert(len(self.manual_bandwidth) == particles.shape[-1])

    #                 # Create the bandwidth array
    #                 bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), device=particles.device)
    #                 for i in range(len(self.manual_bandwidth)):
    #                     bandwidths[...,i] = self.manual_bandwidth[i]

    #             elif(self.manual_bandwidth is not None):
    #                 bandwidths = torch.full(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)


    #         else:
    #             print("Need a bandwidth from somewhere?!?!?!")
    #             assert(False)


    #         # Extract data we need for the frame number we need
    #         particles = particles[frame_number]
    #         particle_weights = particle_weights[frame_number]
    #         bandwidths = bandwidths[frame_number]

    #         # Add a text label Saying what the bandwidths ares
    #         if(is_panel == False):
    #             text_str = "Bandwidths: {:.4f}, {:.4f}, {:.4f}".format(bandwidths.squeeze()[0], bandwidths.squeeze()[1], bandwidths.squeeze()[2])
    #             ax.text(0.55, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)



    #         # # HACK HACK HACK
    #         # # Make sure the bandwidth difference isnt too different
    #         # band_diff = torch.abs(bandwidths[0, 0] - bandwidths[0, 1])
    #         # assert(band_diff <= 0.01)
    #         # band_avg = (bandwidths[0, 0] + bandwidths[0, 1]) / 2.0
    #         # assert(band_avg >= 0.5)
    #         # # sns.kdeplot(x=particles[0,:,0].cpu().numpy(), y=particles[0,:,1].cpu().numpy(), weights=particle_weights[0,:].cpu().numpy(), bw_method=band_avg.cpu().item(), ax=ax, shade=True)
    #         # sns.kdeplot(x=particles[0,:,0].cpu().numpy(), y=particles[0,:,1].cpu().numpy(), weights=particle_weights[0,:].cpu().numpy(), bw_method=band_avg.cpu().item(), ax=ax, shade=True)


    #         # Create the sampling mesh grid
    #         density = 400
    #         x = torch.linspace(x_lim[0], x_lim[1], density)
    #         y = torch.linspace(y_lim[0], y_lim[1], density)
    #         grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    #         points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

    #         # Create the KDE and compute the probs
    #         kde_params_copy = copy.deepcopy(self.kde_params)
    #         del kde_params_copy["dims"][2]
    #         kde = KernelDensityEstimator(kde_params_copy, particles[:, :,0:2], particle_weights, bandwidths[:, 0:2])
    #         log_probs = kde.log_prob(points.unsqueeze(0).to(particles.device)).squeeze(0)
    #         probs = torch.exp(log_probs).cpu()

    #         # Scale
    #         probs_min = torch.min(probs)
    #         probs_max = torch.max(probs)
    #         probs -= probs_min
    #         probs /= (probs_max - probs_min)
    #         probs = probs.reshape(grid_x.shape)



    #         # img = np.zeros((density, density, 3))
    #         # img[...] = 255.0

    #         # for i in range(density):
    #         #     for j in range(density):
    #         #         c = probs[i, j].item()
    #         #         img[i, j, 0:2] = 255.0 - (c*255.0)

    #         # # ax.imshow(img.astype("int"), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='none')
    #         # img = np.transpose(img, axes=[1,0, 2])
    #         # ax.imshow(img.astype("int"), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower")

    #         probs = np.transpose(probs, axes=[1,0])
    #         # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap="BuPu")
    #         # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap="Blues")
    #         # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.color_palette("rocket_r", as_cmap=True))
    #         ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True))




    #         # # Use a KDE to generate samples that we will use in the histogram
    #         # kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
    #         # samples = kde.sample((100000, ))

    #         # # Draw the probability distribution of where we think we should put mass
    #         # x_samples = samples[0, :, 0].cpu().numpy()
    #         # y_samples = samples[0, :, 1].cpu().numpy()
    #         # ranges = [self.evaluation_dataset.get_x_range(), self.evaluation_dataset.get_y_range()]
    #         # ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")
    #         # # ax.hist2d(x_samples, y_samples, range=ranges, bins=20, cmap="Blues")







    #     # If we have particles then draw the particles
    #     if(self.model.outputs_particles_and_weights() and self.render_particles):


    #         # Extract the data that we need
    #         particles = output_dicts["particles"]
    #         particles = particles[frame_number][0]

    #         # Extract the particle weights with the prefix, otherwise just use the other weights
    #         if("{}particle_weights".format(key_name_prefix) in output_dicts):
    #             particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
    #         else:
    #             particle_weights = output_dicts["particle_weights"] 
    #         particle_weights = particle_weights[frame_number]

    #         # Draw the particles
    #         x = particles[:, 0].cpu().numpy()
    #         y = particles[:, 1].cpu().numpy()
    #         particles_scatter = ax.scatter(x,y, color="black", s=0.25*rendered_object_base_size)

    #         # Compute the mean particle
    #         particles = self.model.particle_transformer.backward_tranform(particles)
    #         predicted_state = torch.sum(particles * particle_weights[0].unsqueeze(-1), dim=0)
    #         predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

    #         # Render the mean particle
    #         x = predicted_state[0].item()
    #         y = predicted_state[1].item()
    #         mean_particle_circle = plt.Circle((x, y), 0.25*rendered_object_base_size, color=sns.color_palette("bright")[8])
    #         ax.add_patch(mean_particle_circle)
    #         dy = torch.sin(predicted_state[2]).item() * 1.0*rendered_object_base_size
    #         dx = torch.cos(predicted_state[2]).item() * 1.0*rendered_object_base_size
    #         mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.15*rendered_object_base_size)


    #     if(self.model.outputs_single_solution()):

    #         # Extract the predicted state
    #         predicted_state = output_dicts["predicted_state"]

    #         # Plot the true location
    #         x = predicted_state[frame_number,0,0].item()
    #         y = predicted_state[frame_number,0,1].item()
    #         true_state_circle = plt.Circle((x, y), 0.25 * rendered_object_base_size, color='blue')
    #         ax.add_patch(true_state_circle)
    #         dy = torch.sin(predicted_state[frame_number,0,2]).item() * 1.0 * rendered_object_base_size
    #         dx = torch.cos(predicted_state[frame_number,0,2]).item() * 1.0 * rendered_object_base_size
    #         ax.arrow(x, y, dx, dy, color='blue', width=0.15*rendered_object_base_size)


    #     # Extract the data we need for rendering
    #     observations = data["observations"]
    #     states = data["states"]

    #     # Transform the observations
    #     transformed_observation = self.problem.observation_transformer.forward_tranform(observations)

    #     # Draw the sensor information 
    #     sensors = self.evaluation_dataset.get_sensors()
    #     for i, sensor in enumerate(sensors):

    #         # The sensor location
    #         position = sensor.get_position()
    #         x = position[0].item()
    #         y = position[1].item()
    #         ax.add_patch(plt.Circle((x, y), 0.5*rendered_object_base_size, color='tab:green'))

    #         # The sensor observation
    #         sensor_obs_y = transformed_observation[0, frame_number, i].item() * 20
    #         sensor_obs_x = transformed_observation[0, frame_number, int(transformed_observation.shape[-1] / 2) + i].item() * 20
    #         x_values = [x, sensor_obs_x]
    #         y_values = [y, sensor_obs_y]
    #         ax.plot(x_values, y_values, color="tab:green", linewidth=rendered_object_base_size*1.0)


    #     # Plot the true location
    #     true_x = states[0,frame_number,0].item()
    #     true_y = states[0,frame_number,1].item()
    #     true_state_circle = plt.Circle((true_x, true_y), 0.25 * rendered_object_base_size, color=sns.color_palette("bright")[3])
    #     ax.add_patch(true_state_circle)
    #     dy = torch.sin(states[0,frame_number,2]).item() * 1.0 * rendered_object_base_size
    #     dx = torch.cos(states[0,frame_number,2]).item() * 1.0 * rendered_object_base_size
    #     true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.15*rendered_object_base_size)

    #     if(is_panel):
    #         # Add a text label indicating what frame number we are at
    #         text_str = "Time-step #{:02d}".format(frame_number)
    #         ax.text(0.5, 0.93, text_str, horizontalalignment="center", verticalalignment="center",transform=ax.transAxes, weight="bold", fontsize=12)

    #         return true_state_arrow, mean_particle_arrow, particles_scatter


    #     else:
    #         # Add a text label indicating what frame number we are at
    #         text_str = "Step #{:02d}".format(frame_number)
    #         ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
    #         ax.set_title(title)



    # def draw_angle_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

    #     # Set the X limits for the full unit circle
    #     ax.set_xlim([-np.pi, np.pi])
    #     ax.set_ylim([0, 1.0])



    #     # If we have the KDE then do it
    #     if(self.model.outputs_kde() or (self.model.outputs_particles_and_weights() and self.use_manual_bandwidth)):

    #         # Extract the data that we need
    #         particles = output_dicts["particles"]
    #         # Extract the particle weights with the prefix, otherwise just use the other weights
    #         if("{}particle_weights".format(key_name_prefix) in output_dicts):
    #             particle_weights = output_dicts["{}particle_weights".format(key_name_prefix)] 
    #         else:
    #             particle_weights = output_dicts["particle_weights"] 

    #         # If we are outputting a KDE then we have a bandwidth
    #         if(self.model.outputs_kde()):
    #             # Extract the bandwidths with the prefix, otherwise just use the other bandwidth
    #             if("{}bandwidths".format(key_name_prefix) in output_dicts):
    #                 bandwidths = output_dicts["{}bandwidths".format(key_name_prefix)] 
    #             else:
    #                 bandwidths = output_dicts["bandwidths"] 

    #         elif(self.use_manual_bandwidth and (self.manual_bandwidth is not None)):

    #             # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
    #             if(isinstance(self.manual_bandwidth, list)):

    #                 # Make sure they specified the correct number of bandwidths
    #                 assert(len(self.manual_bandwidth) == particles.shape[-1])

    #                 # Create the bandwidth array
    #                 bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), device=particles.device)
    #                 for i in range(len(self.manual_bandwidth)):
    #                     bandwidths[...,i] = self.manual_bandwidth[i]

    #             elif(self.manual_bandwidth is not None):
    #                 bandwidths = torch.full(size=(particles.shape[0], particles.shape[1], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)


    #         else:
    #             print("Need a bandwidth from somewhere?!?!?!")
    #             assert(False)


    #         # Extract data we need for the frame number we need
    #         particles = particles[frame_number]
    #         particle_weights = particle_weights[frame_number]
    #         bandwidths = bandwidths[frame_number]

    #         # Use a KDE to generate samples that we will use in the histogram
    #         kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

    #         # compute the log prob for samples over the unit circle
    #         x = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(0).unsqueeze(-1).to(self.device)
    #         log_probs = kde.marginal_log_prob(x, [0,1])

    #         # Convert log_probs from log space to normal space
    #         probs = torch.exp(log_probs)

    #         # Plot the probabilities
    #         ax.plot(x.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy(), label="Angle PDF Value")

    #         ax.set_ylim([0, torch.max(probs).item()])

    #     # If we have particles then draw the particles
    #     if(self.model.outputs_particles_and_weights() and self.render_particles):

    #         # Extract the data that we need
    #         particles = output_dicts["particles"]
    #         particles = particles[frame_number]

    #         # Draw the particles
    #         thetas = particles[0, :, 2].cpu().numpy()
    #         ax.vlines(thetas, 0, 0.05, color="black", linewidth=1, label="Particles")

    #     # If we output a single solution
    #     if(self.model.outputs_single_solution()):

    #         # Extract the predicted state
    #         predicted_state = output_dicts["predicted_state"]

    #         # Plot the estimated location
    #         theta = predicted_state[frame_number,0,2].item()
    #         theta = np.asarray(theta)
    #         ax.vlines(theta, 0, 1, color="blue", linewidth=1, label="Predicted Theta")


    #     # Draw a vertical line where the true state is
    #     states = data["states"]
    #     true_theta = states[0, frame_number,2].item()
    #     true_theta = np.asarray(true_theta)
    #     ax.vlines(true_theta, 0, 1, color="red", linewidth=1, label="True Theta")


    #     # Add a text label indicating what frame number we are at
    #     text_str = "Step #{:02d}".format(frame_number)
    #     ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

    #     ax.legend(loc="lower right")
    #     ax.set_title(title)