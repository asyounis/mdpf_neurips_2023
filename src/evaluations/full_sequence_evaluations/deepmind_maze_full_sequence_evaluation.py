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

class DeepmindMazeFullSequenceEvaluation(FullSequenceEvaluationVideoBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)


    def get_rows_and_cols(self):
            
        # 1 - State 
        # 2 - State angle
        # 3 - ESS
        rows = 3
        cols = 2

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
        self.draw_angle_state(frame_number, axes[1,0], data, output_dicts, "State Angle Output")
        

        # Render the ESS if we have the weights
        if(particle_weights is not None):
            self.add_ess_to_plot(axes[2,0], particle_weights, "State Output")

        # If we are resampling then we also want to render the Resampling ESS
        if(self.model.decouple_weights_for_resampling or self.model.decouple_bandwidths_for_resampling):

            # Render the xy state
            self.draw_xy_state(frame_number, axes[0,1], data, output_dicts, "Resampling","resampling_")

            # Render the angle
            self.draw_angle_state(frame_number, axes[1,1], data, output_dicts, "State Angle Resampling", "resampling_")

            # Extract stuff if we have them
            resampling_particle_weights = self.extract_if_present("resampling_particle_weights", output_dicts)

            # If we have the weights then display the ESS
            if(resampling_particle_weights is not None):
                self.add_ess_to_plot(axes[2, 1], resampling_particle_weights, "Resampling")

            # Draw the full path
            self.draw_full_xy_path(frame_number, axes[0, 2], data, output_dicts, "State Output")

        else:
            # Draw the full path
            self.draw_full_xy_path(frame_number, axes[0, 1], data, output_dicts, "State Output")

    # def render_panel(self, render_index, data, output_dicts):

    #     # Get the sequence index, if one doesnt exist then just use the render index
    #     sequence_index = render_index
    #     if("dataset_index" in data):
    #         sequence_index = data["dataset_index"][0]

    #     # Make the figure
    #     rows = self.render_panel_num_rows
    #     cols = self.render_panel_num_cols
    #     number_to_render = (rows * cols) - 1
    #     fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(6*cols, 3*rows))
    #     axes = axes.reshape(-1,)

    #     # Draw all the timesteps
    #     frame_number = 0 
    #     max_rendered_frame_number = 0
    #     for i in range(number_to_render):
    #         frame_number += self.render_panel_modulo    
    #         max_rendered_frame_number = max(frame_number, max_rendered_frame_number)

    #         ax = axes[i]
    #         self.draw_xy_state(frame_number, ax, data, output_dicts, is_panel=True)

    #     # Draw the final fill maze path 
    #     self.draw_full_xy_path(frame_number, axes[-1], data, output_dicts, is_panel=True, max_rendering_index=max_rendered_frame_number)

    #     # Turn off the axis labels
    #     for ax in axes:
    #         ax.axis('off')


    #     # Adjust whitespace
    #     fig.subplots_adjust(wspace=0.03, hspace=0.03)
    #     fig.tight_layout()

    #     # Save the figure as a png
    #     save_file = "{}/panel_rendering_{}.png".format(self.save_dir, render_index)
    #     fig.savefig(save_file)

    #     # Save the figure as a pdf
    #     save_file = "{}/panel_rendering_{}.pdf".format(self.save_dir, render_index)
    #     fig.savefig(save_file, format="pdf")


    def render_panel(self, render_index, data, output_dicts):

        # Get the sequence index, if one doesnt exist then just use the render index
        sequence_index = render_index
        if("dataset_index" in data):
            sequence_index = data["dataset_index"][0]

        # Make the figure
        rows = self.render_panel_num_rows
        cols = self.render_panel_num_cols
        number_to_render = (rows * cols) - 1
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(6*cols, 3*rows))
        axes = axes.reshape(-1,)

        # Draw all the timesteps
        frame_number = 0 
        max_rendered_frame_number = 0
        for i in range(number_to_render):
            frame_number += self.render_panel_modulo    
            max_rendered_frame_number = max(frame_number, max_rendered_frame_number)

            ax = axes[i]
            true_state_arrow, mean_particle_arrow, particles_scatter = self.draw_xy_state(frame_number, ax, data, output_dicts, is_panel=True)

            # Plot the observation as an inset
            # inset_ax = ax.inset_axes([-0.12,0.03,0.6,0.6])
            inset_ax = ax.inset_axes([-0.165,0.03,0.65,0.65])
            observation = torch.permute(data["observations"][0, frame_number].cpu(), (1, 2, 0))
            observation[observation<0] = 0
            observation[observation>1.0] = 1.0
            inset_ax.imshow(observation.numpy())
            inset_ax.set_yticklabels([])
            inset_ax.set_xticklabels([])
            inset_ax.tick_params(left=False, bottom=False)
            inset_ax.patch.set_edgecolor('black')  
            inset_ax.patch.set_linewidth(3)  


        # Draw the final fill maze path 
        self.draw_full_xy_path(frame_number, axes[-1], data, output_dicts, is_panel=True, max_rendering_index=max_rendered_frame_number)

        # Turn off the axis labels
        for ax in axes:
            ax.axis('off')



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
        lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=18)

        # Adjust whitespace
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        # fig.tight_layout()
        # fig.subplots_adjust(wspace=0.01, hspace=0.03)
        fig.tight_layout(rect=(0,0,1,0.94))

        # Save the figure as a png
        save_file = "{}/panel_rendering_{}.png".format(self.save_dir, render_index)
        fig.savefig(save_file)

        # Save the figure as a pdf
        save_file = "{}/panel_rendering_{}.pdf".format(self.save_dir, render_index)
        fig.savefig(save_file, format="pdf")



    def draw_xy_state(self, frame_number, ax, data, output_dicts, title=None, key_name_prefix="", is_panel=False):

        # Extract the map ID
        map_id = data["map_id"].item()

        # Extract the states 
        states = data["states"]

        # Set the X and Y limits
        x_lim = self.evaluation_dataset.get_x_range_scaled(map_id)
        y_lim = self.evaluation_dataset.get_y_range_scaled(map_id)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Draw the walls
        walls = self.evaluation_dataset.get_walls(map_id)
        walls = self.evaluation_dataset.scale_data_down(walls, map_id)
        ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="black", linewidth=3)


        # Compute the base size that we will use scaling all the rendered objects
        x_range = self.evaluation_dataset.get_x_range_scaled(map_id)[1] - self.evaluation_dataset.get_x_range_scaled(map_id)[0]
        y_range = self.evaluation_dataset.get_y_range_scaled(map_id)[1] - self.evaluation_dataset.get_y_range_scaled(map_id)[0]
        max_range = max(x_range, y_range)
        rendered_object_base_size = max_range / 40.0

        if(is_panel):
            rendered_object_base_size *= 2.5



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
            particles = particles[frame_number]
            particle_weights = particle_weights[frame_number]
            bandwidths = bandwidths[frame_number]

            if(is_panel == False):
                # Add a text label Saying what the bandwidths ares
                text_str = "Bandwidths: {:.4f}, {:.4f}, {:.4f}".format(bandwidths.squeeze()[0], bandwidths.squeeze()[1], bandwidths.squeeze()[2])
                ax.text(0.55, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)




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

            # img = np.zeros((density, density, 3))
            # img[...] = 255.0

            # for i in range(density):
            #     for j in range(density):
            #         c = probs[i, j].item()
            #         img[i, j, 0:2] = 255.0 - (c*255.0)

            # # ax.imshow(img.astype("int"), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='none')
            # img = np.transpose(img, axes=[1,0, 2])
            # ax.imshow(img.astype("int"), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower")

            probs = np.transpose(probs, axes=[1,0])
            # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap="BuPu")
            # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap="Blues")
            # ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.color_palette("rocket_r", as_cmap=True))
            ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True), aspect="auto")



            # band_diff = torch.abs(bandwidths[0, 0] - bandwidths[0, 1])
            # if(band_diff <= 0.01):


            #     # HACK HACK HACK
            #     # Make sure the bandwidth difference isnt too different
            #     assert(band_diff <= 0.01)
            #     band_avg = (bandwidths[0, 0] + bandwidths[0, 1]) / 2.0
            #     assert(band_avg >= 0.5)
            #     # sns.kdeplot(x=particles[0,:,0].cpu().numpy(), y=particles[0,:,1].cpu().numpy(), weights=particle_weights[0,:].cpu().numpy(), bw_method=band_avg.cpu().item(), ax=ax, shade=True)
            #     sns.kdeplot(x=particles[0,:,0].cpu().numpy(), y=particles[0,:,1].cpu().numpy(), weights=particle_weights[0,:].cpu().numpy(), bw_method=band_avg.cpu().item(), ax=ax, shade=True)

            # else:

            #     # Use a KDE to generate samples that we will use in the histogram
            #     kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
            #     samples = kde.sample((100000, ))

            #     # Draw the probability distribution of where we think we should put mass
            #     x_samples = samples[0, :, 0].cpu().numpy()
            #     y_samples = samples[0, :, 1].cpu().numpy()
            #     ranges = [self.evaluation_dataset.get_x_range_scaled(map_id), self.evaluation_dataset.get_y_range_scaled(map_id)]
            #     ax.hist2d(x_samples, y_samples, range=ranges, bins=100, cmap="Blues")



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"]
            particle_weights = output_dicts["particle_weights"] 

            # Get the right set of particles.  
            particles = particles[frame_number][0]
            particle_weights = particle_weights[frame_number][0]

            # Draw the particles
            x = particles[:, 0].cpu().numpy()
            y = particles[:, 1].cpu().numpy()
            if(is_panel):
                particles_scatter = ax.scatter(x,y, color="black", s=3.0*rendered_object_base_size)
            else:
                particles_scatter = ax.scatter(x,y, color="black", s=1*rendered_object_base_size)

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
            mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.15*rendered_object_base_size)


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
            ax.text(0.75, 0.05, text_str, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, weight="bold", fontsize=16)

            return true_state_arrow, mean_particle_arrow, particles_scatter

        else:
            # Add a text label indicating what frame number we are at
            text_str = "Step #{:02d}".format(frame_number)
            ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)
            ax.set_title(title)


    def draw_angle_state(self, frame_number, ax, data, output_dicts, title, key_name_prefix=""):

        # Set the X limits for the full unit circle
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([0, 1.0])



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
            ax.plot(x.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy(), label="Angle PDF Value")

            ax.set_ylim([0, torch.max(probs).item()])



        # If we have particles then draw the particles
        if(self.model.outputs_particles_and_weights() and self.render_particles):

            # Extract the data that we need
            particles = output_dicts["particles"]
            particles = particles[frame_number]

            # Draw the particles
            thetas = particles[0, :, 2].cpu().numpy()
            ax.vlines(thetas, 0, 0.05, color="black", linewidth=1, label="Particles")





        # Draw a vertical line where the true state is
        states = data["states"]
        true_theta = states[0, frame_number,2].item()
        true_theta = np.asarray(true_theta)
        ax.vlines(true_theta, 0, 1, color="red", linewidth=1, label="True Theta")


        # Add a text label indicating what frame number we are at
        text_str = "Step #{:02d}".format(frame_number)
        ax.text(0.95, 0.95, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

        ax.legend(loc="lower right")
        ax.set_title(title)


    def draw_full_xy_path(self, frame_number, ax, data, output_dicts, title=None, key_name_prefix="", is_panel=False, max_rendering_index=None):

        # Extract the map ID
        map_id = data["map_id"].item()

        # Extract the states 
        states = data["states"]
        if(max_rendering_index is not None):
            states = states[:,:max_rendering_index, ...]

        # Set the X and Y limits
        ax.set_xlim(self.evaluation_dataset.get_x_range_scaled(map_id))
        ax.set_ylim(self.evaluation_dataset.get_y_range_scaled(map_id))

        # Draw the walls
        walls = self.evaluation_dataset.get_walls(map_id)
        walls = self.evaluation_dataset.scale_data_down(walls, map_id)
        ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="black", linewidth=3)

        if(is_panel == False):

            # If we have particles then draw the particles
            if(self.model.outputs_particles_and_weights()):

                # Extract the data that we need
                particles = output_dicts["particles"]
                particle_weights = output_dicts["particle_weights"] 

                # Compute the mean particle
                particles = self.model.particle_transformer.backward_tranform(particles)
                predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=2)
                predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

                if(max_rendering_index is not None):
                    predicted_state = predicted_state[:max_rendering_index, ...]

                # Draw the Predicted path
                x1 = predicted_state[:,0,0].detach().cpu().numpy()
                y1 = predicted_state[:,0,1].detach().cpu().numpy()
                ax.plot(x1, y1, color="black", label="PF Path")


        # Draw the true path
        x1 = states[0, :, 0].numpy()
        y1 = states[0, :, 1].numpy()
        ax.plot(x1, y1, color="red", marker="o", label="True Path", linewidth=4)

        if(is_panel):
            ax.legend(loc='lower left', fontsize=12)