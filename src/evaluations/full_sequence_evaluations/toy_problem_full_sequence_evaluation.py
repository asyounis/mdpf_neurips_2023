# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *

# Project imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_image_base import *
from models.sequential_models import *

class ToyProblemFullSequenceEvaluation(FullSequenceEvaluationImageBase):
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
            
        # 1 - State 
        # 2 - ESS
        rows = 4
        cols = 1

        if(self.model.decouple_weights_for_resampling):

            # 1 - Resampling State 
            # 2 - Resampling ESS
            rows += 2

        return rows, cols


    def extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None


    def render_data(self, data, output_dicts, axes):

        # Flatten the axis to make life easier (aka Im lazy)
        axes = np.reshape(axes, (-1, ))

        # Extract the data 
        states = data["states"]
        start_time = data["start_time"]
        observations = data["observations"]

        # Extract stuff if we have them        
        particles = self.extract_if_present("particles", output_dicts)
        particle_weights = self.extract_if_present("particle_weights", output_dicts)
        bandwidths = self.extract_if_present("bandwidths", output_dicts)
        resampling_particle_weights = self.extract_if_present("resampling_particle_weights", output_dicts)
        resampling_bandwidths = self.extract_if_present("resampling_bandwidths", output_dicts)

        # Extract useful stats
        subsequence_length = states.shape[1]

        # Extract the min and max of the axis
        axis_range = self.evaluation_dataset.get_range()
        axis_min = axis_range[0]
        axis_max = axis_range[1]

        # Extract some stuff we want for plotting
        plotting_steps = np.arange(0, subsequence_length) + 0.5

        # Extract the single state solutions from the KDE
        most_probable_states, most_probable_particles = self.extract_single_state_solution(particles, particle_weights, bandwidths)

        ax = axes[0]
        ax.set_ylim(axis_min, axis_max)
        ax.plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
        ax.plot(plotting_steps, most_probable_states[:,0].cpu().numpy(), label="Most Probable KDE State", color="blue")
        ax.plot(plotting_steps, most_probable_particles[:,0].cpu().numpy(), label="Most Particle", color="orange")
        ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
        ax.set_title("State Predictions")
        ax.set_xlabel("Time-step")
        ax.set_ylabel("State")
        ax.legend()

        # Render the HMM probability image
        ax = axes[1]
        hmm_image_data = self.create_hmm_image(states, observations, start_time, axis_min, axis_max)
        ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
        ax.plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
        ax.imshow(hmm_image_data, extent=[0, hmm_image_data.shape[1], axis_min, axis_max], aspect="auto", interpolation="none",cmap=plt.get_cmap("Greys"))
        ax.set_title("Probability Densities HMM")
        ax.set_xlabel("Time-step")
        ax.set_ylabel("State")
        ax.set_ylim(axis_min, axis_max)
        ax.legend()

        if(self.model.outputs_kde()):

            # Render the KDE probability image 
            ax = axes[2]
            kde_image_data = self.create_kde_image(particles, particle_weights, bandwidths)
            ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
            ax.plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
            ax.imshow(kde_image_data, extent=[0, kde_image_data.shape[1], axis_min, axis_max], aspect="auto", interpolation="none",cmap=plt.get_cmap("Greys"))
            ax.set_title("Probability Densities KDE")
            ax.set_xlabel("Time-step")
            ax.set_ylabel("State")
            ax.set_ylim(axis_min, axis_max)
            ax.legend()

            increment = 1.0 / float(bandwidths.shape[0])
            for i in range(0,bandwidths.shape[0], 5):
                b = bandwidths[i].squeeze().item()

                # Add a text label Saying what the bandwidths are
                text_str = "{:.3f}".format(b)
                plt.rcParams.update({'font.size': 8})
                ax.text(float(i) * increment + increment/2, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)

        if(self.model.outputs_particles_and_weights()):

            flat_particles = particles[:,0, :, 0]
            plotting_steps_2 = torch.tile(torch.from_numpy(plotting_steps).unsqueeze(-1), [1, flat_particles.shape[1]])

            # Compute the colors
            max_weight = torch.max(particle_weights[:,0,:], dim=1, keepdim=True)[0]
            colors = particle_weights / max_weight.unsqueeze(-1)

            # Apply a min color value
            min_color_value = 0.15
            colors[colors < min_color_value] = min_color_value

            # Render the particles 
            ax = axes[2]

            ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
            ax.plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
            # ax.scatter(plotting_steps_2.cpu().numpy(),flat_particles.cpu().numpy(), c=colors.cpu().numpy(), cmap="Blues", label="Particles",  s=5.0)
            ax.scatter(plotting_steps_2.cpu().numpy(),flat_particles.cpu().numpy(), color="red", label="Particles",  s=5.0)
            ax.set_title("Particles")
            ax.set_xlabel("Time-step")
            ax.set_ylabel("State")
            ax.set_ylim(axis_min, axis_max)
            ax.legend()



        # Render the resampling KDE probability
        if(self.model.decouple_weights_for_resampling):
            ax = axes[3]
            kde_image_data = self.create_kde_image(particles, resampling_particle_weights, resampling_bandwidths)
            ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
            ax.plot(plotting_steps, states[0, :,0].cpu().numpy(), label="True State", color="green")
            ax.imshow(kde_image_data, extent=[0, kde_image_data.shape[1], axis_min, axis_max], aspect="auto", interpolation="none",cmap=plt.get_cmap("Greys"))
            ax.set_title("Probability Densities KDE (Resampling)")
            ax.set_xlabel("Time-step")
            ax.set_ylabel("State")
            ax.set_ylim(axis_min, axis_max)
            ax.legend()

            increment = 1.0 / float(resampling_bandwidths.shape[0])
            for i in range(resampling_bandwidths.shape[0]):
                b = resampling_bandwidths[i].squeeze().item()

                # Add a text label Saying what the resampling_bandwidths are
                text_str = "{:.3f}".format(b)
                plt.rcParams.update({'font.size': 8})
                ax.text(float(i) * increment + increment/2, 0.90, text_str, horizontalalignment='center', verticalalignment='center',transform = ax.transAxes)





        # Render the ESS if we have the weights
        if(particle_weights is not None):
            if(resampling_particle_weights is not None):
                self.add_ess_to_plot(axes[-2], particle_weights, "State Output", plotting_steps)
            else:
                self.add_ess_to_plot(axes[-1], particle_weights, "State Output", plotting_steps)

        # If we have the weights then display the ESS
        if(resampling_particle_weights is not None):
            self.add_ess_to_plot(axes[-1], resampling_particle_weights, "Resampling", plotting_steps)


    def add_ess_to_plot(self, ax, weights, weights_for, plotting_steps):

        # Get some stats
        subsequence_length = weights.shape[0]
        number_of_particles = weights.shape[2]

        # Compute the effective sample size
        ess = weights.squeeze(1)**2
        ess = torch.sum(ess, dim=-1)
        ess = 1.0 / ess

        # Extract the min and max of this plot so we can add the verticle line
        axis_min = 0
        axis_max = torch.max(ess).item()

        # Plot the ess
        ax.plot(plotting_steps, ess.cpu().numpy(), label="Effective Sample Size", color="black")
        ax.vlines(plotting_steps, axis_min, axis_max, color="red", linewidth=0.5)
        ax.set_title("Effective Sample Size State (N={}) for {}".format(number_of_particles, weights_for))
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Effective Sample Size")


    def extract_single_state_solution(self, particles, particle_weights, bandwidths):
        
        # Extract some stats
        subsequence_length = particles.shape[0]

        if(bandwidths is not None):


            # Create some test points that we will use for creating the image
            states_range = self.evaluation_dataset.get_range()
            test_points = torch.linspace(states_range[0], states_range[1], 100000).unsqueeze(0).unsqueeze(-1).to(particles.device)

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


    def create_kde_image(self, particles, particle_weights, bandwidths):

        # Extract some stats
        subsequence_length = particles.shape[0]

        # Create some test points that we will use for creating the image
        states_range = self.evaluation_dataset.get_range()
        test_points = torch.linspace(states_range[0], states_range[1], 100000).unsqueeze(0).unsqueeze(-1).to(particles.device)

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


    def create_hmm_image(self, states, observations, start_time ,state_min, state_max, number_of_discrete_states=1000):

        # Move things to the GPU
        states = states.to(self.device)
        observations = observations.to(self.device)
        start_time = start_time.to(self.device)


        # The discretization that we will use for the x_{t-1} and x_t states
        state_discretization = torch.linspace(state_min, state_max, number_of_discrete_states, device=self.device)

        # Constants from the paper
        SIGMA_U = math.sqrt(10)
        SIGMA_V = math.sqrt(1)

        # Normal distributions for calculating the probs.  Use a mean 0 dist and then 
        # Adjust the data to use mean 0. 
        state_normal = D.Normal(torch.tensor(0).to(self.device), torch.tensor(SIGMA_U).to(self.device))
        obs_normal = D.Normal(torch.tensor(0).to(self.device), torch.tensor(SIGMA_V).to(self.device))

        # All the probabilities for the stat
        all_state_probs = torch.zeros(size=(states.shape[0], states.shape[1], state_discretization.shape[0]), device=self.device)

        # Set the starting state to the correct value aka the discretized state closes to the true state
        starting_state = states[:,0,0]
        for i in range(states.shape[0]):
            diff = torch.abs(state_discretization - starting_state[i])
            start_state_idx = torch.argmin(diff)
            all_state_probs[i,0,start_state_idx.cpu().item()] = 1.0

        # Compute the x_means for all the states
        x_mean_from_prev_state_no_cos = (state_discretization / 2.0)
        x_mean_from_prev_state_no_cos += 25.0 * (state_discretization / (1 + (state_discretization**2)))

        # Run the HMM
        for i in range(1, states.shape[1]):

            # Finish off making the mean by adding in the cos term
            current_time = start_time.clone() + i
            x_mean = torch.tile(x_mean_from_prev_state_no_cos.unsqueeze(0),[states.shape[0], 1])
            x_mean += (8 * torch.cos(1.2 * current_time))

            # Compute the probability from the previous states to the current state
            # Do it this was so that we can use 1 normal dist that is on the GPU instead 
            # of constantly creating a new one on the GPU which is ~2x slower        
            prob_states = torch.tile(state_discretization.unsqueeze(1),[1, state_discretization.shape[0]])
            prob_states = torch.tile(prob_states.unsqueeze(0),[states.shape[0], 1, 1])
            prob_states -= x_mean.unsqueeze(1)
            prob_states = state_normal.log_prob(prob_states)
            prob_states = torch.exp(prob_states)

            # Compute the probability of the observation given the current state
            obs_means = (state_discretization**2) / 20.0
            obs_means = torch.tile(obs_means.unsqueeze(0), [states.shape[0], 1])
            obs_probs = -obs_means + observations[:,i]
            obs_probs = obs_normal.log_prob(obs_probs)
            obs_probs = torch.exp(obs_probs)

            for state_idx  in range(state_discretization.shape[0]):

                # Compute the forward value for this state at this time step
                tmp = all_state_probs[:,i-1] * prob_states[:, state_idx]
                tmp = torch.sum(tmp, dim=-1)
                tmp *= obs_probs[:,state_idx]

                all_state_probs[:, i, state_idx] = tmp


            # Normalize the messages
            normalizer = torch.sum(all_state_probs[:,i,:], dim=-1)    
            all_state_probs[:, i, :] /= (normalizer.unsqueeze(-1) + 1e-8)

        # Get rid of the batch dim
        all_state_probs = all_state_probs.squeeze(0)


        # Generate the hmm likelihood Image
        hmm_image_data = all_state_probs
        mins,_ = torch.min(hmm_image_data,dim=-1, keepdim=True)
        maxs,_ = torch.max(hmm_image_data,dim=-1, keepdim=True)
        hmm_image_data = hmm_image_data - mins
        hmm_image_data = hmm_image_data / (maxs-mins)
        hmm_image_data = hmm_image_data.cpu().numpy().transpose()
        hmm_image_data = np.flip(hmm_image_data, axis=0)



        # Return the state probs.  We remove the first step since that is the starting state
        return hmm_image_data
