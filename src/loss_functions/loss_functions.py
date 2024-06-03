# Project Imports
from loss_functions.loss_function_base import *
from loss_functions.bounding_box_loss_functions import *

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *

import time



class KDE_NLL(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)

        # Extract the KDE params that we will be using to compute this loss fuction
        assert("kde_params" in loss_params)
        self.kde_params = loss_params["kde_params"]

        # If we have a manual bandwidth set then we need use it
        if("manual_bandwidth" in loss_params):
            self.manual_bandwidth = loss_params["manual_bandwidth"]
        else:
            self.manual_bandwidth = None

        # Create the kde computer
        self.kde_computer = KDEComputer(self.kde_params)

        # Cache the bandwidths if we can to make things faster
        self.cached_manual_bandwidths = None

    def compute_loss(self, output_dict, states):
        
        # Unpack the output dict to get what we need for the loss

        # Check what kinda of data we have
        if("particles" in output_dict):
            particles = output_dict["particles"]
            particle_weights = output_dict["particle_weights"]
        elif("predicted_state" in output_dict):

            # Set the particles to be this 1 predicted state.  Use uniform weights
            particles = output_dict["predicted_state"].unsqueeze(1)
            particle_weights = torch.ones((particles.shape[0], 1), device=particles.device)
        else:
            assert(False)    

        # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
        if(isinstance(self.manual_bandwidth, list)):

            # Make sure they specified the correct number of bandwidths
            assert(len(self.manual_bandwidth) == len(self.kde_computer.distribution_types))

            if((self.cached_manual_bandwidths is None) or (self.cached_manual_bandwidths.shape[0] != particles.shape[0])):
                # Create the bandwidth array
                self.cached_manual_bandwidths = torch.zeros(size=(particles.shape[0], len(self.manual_bandwidth)), device=particles.device)
                for i in range(len(self.manual_bandwidth)):
                    self.cached_manual_bandwidths[...,i] = self.manual_bandwidth[i]

            bandwidths = self.cached_manual_bandwidths

        elif(self.manual_bandwidth is not None):
            if((self.cached_manual_bandwidths is None) or (self.cached_manual_bandwidths.shape[0] != particles.shape[0]) or (self.cached_manual_bandwidths.shape[1] != particles.shape[-1])):
                self.cached_manual_bandwidths = torch.full(size=(particles.shape[0], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)

            bandwidths = self.cached_manual_bandwidths
        else:
            assert("bandwidths" in output_dict)
            bandwidths = output_dict["bandwidths"]


        # Compute the loss function
        loss = self.kde_computer.compute_kde(states.unsqueeze(1), particles, particle_weights, bandwidths)
        loss = loss + 1e-8
        loss = -torch.log(loss)

        return loss


class DistanceError(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)


        if("angle_dims" in loss_params):
            self.angle_dims = loss_params["angle_dims"]

            # Convert to a set for faster lookup
            self.angle_dims = set(self.angle_dims)
        else:
            self.angle_dims = set()



        if("ignore_dims" in loss_params):
            self.ignore_dims = loss_params["ignore_dims"]

            # Convert to a set for faster lookup
            self.ignore_dims = set(self.ignore_dims)
        else:
            self.ignore_dims = set()


        # The ranges to use when normalizing the outputs
        if("ranges" in loss_params):
            self.ranges = loss_params["ranges"] 
        else:
            self.ranges = dict()

        # The dims to select for
        if("selected_dims" in loss_params):
            self.selected_dims = loss_params["selected_dims"] 
        else:
            self.selected_dims = None

        # The rescale to amount to use when normalizing the outputs
        if("rescaled_ranges" in loss_params):
            self.rescaled_ranges = loss_params["rescaled_ranges"] 
        else:
            self.rescaled_ranges = dict()

        # What types of outputs we should use when we are
        if("output_select" in loss_params):
            self.output_select = loss_params["output_select"]

            # Make sure that the output selected is a valid value
            assert((self.output_select == "auto") or (self.output_select == "predicted_state") or (self.output_select == "particles") or (self.output_select == "peak_finding") or (self.output_select == "pf_map"))

            # If we have peak finding then we need a bandwidth
            if(self.output_select == "peak_finding"):        
                # If we have a manual bandwidth set then we need use it
                if("manual_bandwidth" in loss_params):
                    self.manual_bandwidth = loss_params["manual_bandwidth"]
                else:
                    self.manual_bandwidth = None

                # Cache the bandwidths if we can to make things faster
                self.cached_manual_bandwidths = None

        else:
            self.output_select = "auto"    

    def compute_loss(self, output_dict, states):

        # figure out what to use
        if(self.output_select == "auto"):
            if("particles" in output_dict):
                selection = "particles"    
            elif("predicted_state" in output_dict):
                selection = "predicted_state"
            else:
                assert(False)
        # elif(self.output_select == "predicted_state"):
        #     selection = "predicted_state"
        # elif(self.output_select == "particles"):
        #     selection = "particles"
        # elif(self.output_select == "peak_finding"):
        #     selection = "peak_finding"
        # elif(self.output_select == "pf_map"):
        #     selection = "pf_map"
        else:
            assert(False)        

        # If we have particles then we need to 
        if(selection == "particles"):
            # Unpack the output dict to get what we need for the loss
            particles = output_dict["particles"]
            particle_weights = output_dict["particle_weights"]

            # convert to the higher space
            particles = self.model.particle_transformer.backward_tranform(particles)

            # Compute the mean 
            predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=-2)

            # Convert back
            predicted_state = self.model.particle_transformer.forward_tranform(predicted_state)

        elif(selection == "predicted_state"):
            predicted_state = output_dict["predicted_state"].squeeze(1)

        elif(selection == "peak_finding"):

            # Unpack the output dict to get what we need for the loss
            particles = output_dict["particles"]
            particle_weights = output_dict["particle_weights"]

            # If we have a manual bandwidth then use it otherwise there better be a bandwidth in the output dict
            if(isinstance(self.manual_bandwidth, list)):

                # Make sure they specified the correct number of bandwidths
                # assert(len(self.manual_bandwidth) == len(self.kde_computer.distribution_types))

                if((self.cached_manual_bandwidths is None) or (self.cached_manual_bandwidths.shape[0] != particles.shape[0])):
                    # Create the bandwidth array
                    self.cached_manual_bandwidths = torch.zeros(size=(particles.shape[0], len(self.manual_bandwidth)), device=particles.device)
                    for i in range(len(self.manual_bandwidth)):
                        self.cached_manual_bandwidths[...,i] = self.manual_bandwidth[i]

                bandwidths = self.cached_manual_bandwidths

            elif(self.manual_bandwidth is not None):
                if((self.cached_manual_bandwidths is None) or (self.cached_manual_bandwidths.shape[0] != particles.shape[0]) or (self.cached_manual_bandwidths.shape[1] != particles.shape[-1])):
                    self.cached_manual_bandwidths = torch.full(size=(particles.shape[0], particles.shape[-1]), fill_value=self.manual_bandwidth, device=particles.device)

                bandwidths = self.cached_manual_bandwidths
            else:
                assert("bandwidths" in output_dict)
                bandwidths = output_dict["bandwidths"]

            # Create the KDE
            kde = KernelDensityEstimator(self.model.kde_params, particles, particle_weights, bandwidths)

            # Find the peak of the KDE
            predicted_state = kde.find_peak()

        elif(selection == "pf_map"):

            # Unpack the output dict to get what we need for the loss
            particles = output_dict["particles"]
            particle_weights = output_dict["particle_weights"]

            # Select the highest probability sample
            highest_prob_particle = torch.argmax(particle_weights, dim=-1)
            predicted_state = torch.zeros((particles.shape[0], particles.shape[-1]), device=particles.device)
            for i in range(particles.shape[0]):
                predicted_state[i, :] = particles[i, highest_prob_particle[i], :]

        # dataset_obj = output_dict["dataset_object"]
        # states = dataset_obj.scale_data_up(states)
        # predicted_state = dataset_obj.scale_data_up(predicted_state)

        # compute the squared errors
        all_errors = []
        for i in range(states.shape[-1]):

            state = states[...,i]
            ps = predicted_state[...,i]

            # Only use the selected dims
            if(self.selected_dims is not None):
                if(i not in self.selected_dims):
                    continue

            # rescale to [0, 1] if we have the range set
            if(i in self.ranges):
                min_val = self.ranges[i]["min"]
                max_val = self.ranges[i]["max"]

                # Scale prediction
                ps = ps - min_val
                ps = ps / (max_val - min_val)

                # Scale the true state
                state = state - min_val
                state = state / (max_val - min_val)


            # rescale.  This assumes the predicted state is already in the range of [0, 1]
            if(i in self.rescaled_ranges):
                min_val = self.rescaled_ranges[i]["min"]
                max_val = self.rescaled_ranges[i]["max"]

                # Scale prediction
                ps = ps * (max_val - min_val)
                ps = ps + min_val
                    
                # Scale the true state
                state = state * (max_val - min_val)
                state = state + min_val


            if(i in self.ignore_dims):
                continue

            # Make sure we handle angles well
            if(i in self.angle_dims):
                error = self.compute_angle_error(ps, state)
            else:
                error = self.compute_error(ps, state)

            all_errors.append(error)
        all_errors = torch.stack(all_errors, dim=-1)

        return all_errors

    def compute_error(self, predicted_state, state):
        raise NotImplemented

    def compute_angle_error(self, predicted_state, state):
        raise NotImplemented

    def compute_angle_difference(self, true_angle, predicted_angle):
        rotational_error = (predicted_angle - true_angle + np.pi) % (2 * np.pi)
        rotational_error -= np.pi 
        rotational_error[rotational_error < -np.pi] += 2.0 *np.pi
        
        return rotational_error


class SquaredError(DistanceError):
    def __init__(self, loss_params, model):
        super().__init__(loss_params,model)

    def compute_error(self, predicted_state, state):
        return(predicted_state - state)**2

    def compute_angle_error(self, predicted_state, state):
        return self.compute_angle_difference(state, predicted_state)**2

class MSE(SquaredError):
    def __init__(self, loss_params, model):
        super().__init__(loss_params ,model)

    def compute_loss(self, output_dict, states):
        
        # Get the Squared errors from the parent class
        all_ses = super().compute_loss(output_dict, states)

        # Take the sum to compute the mean squared error
        mse = torch.sum(all_ses, dim=-1)
        
        return mse


# class RMSE(MSE):
#     def __init__(self, loss_params, model):
#         super().__init__(loss_params ,model)

#     def do_final_aggrigation(self, data):


#         print(data.shape)
#         exit()

#         return torch.sqrt(torch.mean(data)).item()


class RMSE(MSE):
    def __init__(self, loss_params, model):
        super().__init__(loss_params ,model)

    def do_final_aggrigation(self, data):
        return torch.sqrt(data)



# class RMSE(SquaredError):
#     def __init__(self, loss_params, model):
#         super().__init__(loss_params ,model)

#     def compute_loss(self, output_dict, states):
        
#         # Get the Squared errors from the parent class
#         all_ses = super().compute_loss(output_dict, states)

#         # Take the mean to compute the mean squared error
#         # mse = torch.mean(all_ses, dim=-1)
#         mse = torch.sum(all_ses, dim=-1)

#         mse = torch.sqrt(mse)
        
#         return mse




class MSESeparateAnglePosition(SquaredError):
    def __init__(self, loss_params, model):
        super().__init__(loss_params, model)

        self.alpha = loss_params["alpha"]


    def compute_loss(self, output_dict, states):
        
        # Get the Squared errors from the parent class
        all_ses = super().compute_loss(output_dict, states)


        position_dims = []
        angle_dims = []
        for i in range(all_ses.shape[-1]):
            if i in self.angle_dims:
                angle_dims.append(i)
            else:
                position_dims.append(i)


        position_ses = all_ses[:, position_dims]
        angle_ses = all_ses[:, angle_dims]

        position_mse = torch.mean(position_ses)
        angle_mse = torch.mean(angle_ses)

        mse = position_mse + (self.alpha * angle_mse)

        return mse




class AbsoluteError(DistanceError):
    def __init__(self, loss_params, model):
        super().__init__(loss_params,model)

    def compute_loss(self, output_dict, states):
        
        # Get the Squared errors from the parent class
        all_ses = super().compute_loss(output_dict, states)

        # Take the mean to compute the mean squared error
        mse = torch.sum(all_ses, dim=-1)
        
        return mse

    def compute_error(self, predicted_state, state):
        return torch.abs(predicted_state - state)

    def compute_angle_error(self, predicted_state, state):
        return torch.abs(self.compute_angle_difference(state, predicted_state))


class ThesholdedDistanceError(DistanceError):
    def __init__(self, loss_params, model):
        super().__init__(loss_params ,model)

        assert("threshold" in loss_params)
        self.threshold = loss_params["threshold"]


    def compute_loss(self, output_dict, states):

        # get the absoulte error
        errors  = super().compute_loss(output_dict, states)
        errors = torch.sum(errors, dim=-1)
        errors = torch.sqrt(errors)
        
        # Compute the out of threshold
        out_of_theshold = torch.zeros(size=(errors.shape[0],), device=errors.device)        
        mask = (errors >= self.threshold)
        out_of_theshold[mask] = 1.0
        
        # Compute accuracy
        # accuracy = 1.0 - out_of_theshold        

        return out_of_theshold


    def compute_error(self, predicted_state, state):
        return (predicted_state - state)**2

    def compute_angle_error(self, predicted_state, state):
        return self.compute_angle_difference(state, predicted_state)**2





class ExpectedThesholdedDistanceError(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)

        assert("threshold" in loss_params)
        self.threshold = loss_params["threshold"]

        assert("number_of_samples_to_draw" in loss_params)
        self.number_of_samples_to_draw = loss_params["number_of_samples_to_draw"]

        # Extract the KDE params if we have them.  These will be used for sampling from KDEs if the bandwidth is provided
        if("kde_params" in loss_params):
            self.kde_params = loss_params["kde_params"]
        else:
            self.kde_params = None



        if("angle_dims" in loss_params):
            self.angle_dims = loss_params["angle_dims"]

            # Convert to a set for faster lookup
            self.angle_dims = set(self.angle_dims)
        else:
            self.angle_dims = set()

        # The ranges to use when normalizing the outputs
        if("ranges" in loss_params):
            self.ranges = loss_params["ranges"] 
        else:
            self.ranges = dict()

        # The dims to select for
        if("selected_dims" in loss_params):
            self.selected_dims = loss_params["selected_dims"] 
        else:
            self.selected_dims = None

        # The rescale to amount to use when normalizing the outputs
        if("rescaled_ranges" in loss_params):
            self.rescaled_ranges = loss_params["rescaled_ranges"] 
        else:
            self.rescaled_ranges = dict()

    def compute_loss(self, output_dict, states):

        # Unpack the output dict to get what we need for the loss
        particles = output_dict["particles"]
        particle_weights = output_dict["particle_weights"]
        bandwidths = output_dict["bandwidths"]

        # Draw samples
        if(bandwidths is None):
            samples = self.sample_discrete(particles, particle_weights)
        else:
            samples = self.sample_kde(particles, particle_weights, bandwidths)

        # compute the squared errors
        all_errors = []
        for i in range(states.shape[-1]):

            # Get the state at this timestep
            state = states[...,i]

            # Only use the selected dims
            if(self.selected_dims is not None):
                if(i not in self.selected_dims):
                    continue

            # rescale to [0, 1] if we have the range set
            if(i in self.ranges):
                min_val = self.ranges[i]["min"]
                max_val = self.ranges[i]["max"]

                # Scale prediction
                samples[...,i] = samples[...,i] - min_val
                samples[...,i] = samples[...,i] / (max_val - min_val)

                # Scale the true state
                state = state - min_val
                state = state / (max_val - min_val)

            # rescale.  This assumes the predicted state is already in the range of [0, 1]
            if(i in self.rescaled_ranges):
                min_val = self.rescaled_ranges[i]["min"]
                max_val = self.rescaled_ranges[i]["max"]

                # Scale prediction
                samples[...,i] = samples[...,i] * (max_val - min_val)
                samples[...,i] = samples[...,i] + min_val
                
                # Scale the true state
                state = state * (max_val - min_val)
                state = state + min_val


            # Make sure we handle angles well
            if(i in self.angle_dims):
                error = self.compute_angle_error(samples[...,i], state)
            else:
                error = self.compute_error(samples[...,i], state)

            all_errors.append(error)

        errors = torch.stack(all_errors, dim=-1)


        errors = torch.sum(errors, dim=-1)
        errors = torch.sqrt(errors)
        
        # Compute the out of threshold
        out_of_theshold = torch.zeros(size=(errors.shape[0],errors.shape[1]), device=errors.device)        
        mask = (errors >= self.threshold)
        out_of_theshold[mask] = 1.0
        
        avg_out_of_theshold = torch.mean(out_of_theshold, dim=1)

        return avg_out_of_theshold


    def sample_discrete(self, particles, particle_weights):
        
        # Need some stats
        batch_size = particles.shape[0]
        device = particles.device

        # Compute the cumulative sum of the particle weights so we can use uniform sampling
        particle_weights_cumsum = torch.cumsum(particle_weights, dim=-1)
        particle_weights_cumsum = torch.tile(particle_weights_cumsum.unsqueeze(1), [1, self.number_of_samples_to_draw, 1])

        # Generate random numbers, use the same random numbers for all batches
        uniform_random_nums = torch.rand(size=(batch_size, self.number_of_samples_to_draw, 1),device=device)

        # Select the particle indices's
        selected = particle_weights_cumsum >= uniform_random_nums
        _, selected = torch.max(selected, dim=-1)

        # Resample
        samples = torch.zeros(size=(batch_size, self.number_of_samples_to_draw, particles.shape[-1]), device=device)
        for b in range(batch_size):
            samples[b,...] = particles[b,selected[b] ,...]
        samples = torch.cat([particles[b,selected[b] ,...].unsqueeze(0) for b in range(batch_size)])

        return samples

    def sample_kde(self, particles, particle_weights, bandwidths):

        # We need the kde params for this sampling method
        if(self.kde_params is None):
            assert(False)

        # Create the KDE
        kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

        # Sample
        return kde.sample((self.number_of_samples_to_draw,))

    def compute_error(self, samples, state):
        return (samples - state.unsqueeze(1))**2

    def compute_angle_error(self, samples, state):
        return self.compute_angle_difference(state.unsqueeze(1), samples)**2

    def compute_angle_difference(self, true_angle, predicted_angle):
        rotational_error = (predicted_angle - true_angle + np.pi) % (2 * np.pi)
        rotational_error -= np.pi 
        rotational_error[rotational_error < -np.pi] += 2.0 *np.pi
        
        return rotational_error




class ExpectedDistanceError(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)

        assert("number_of_samples_to_draw" in loss_params)
        self.number_of_samples_to_draw = loss_params["number_of_samples_to_draw"]

        # Extract the KDE params if we have them.  These will be used for sampling from KDEs if the bandwidth is provided
        if("kde_params" in loss_params):
            self.kde_params = loss_params["kde_params"]
        else:
            self.kde_params = None



        if("angle_dims" in loss_params):
            self.angle_dims = loss_params["angle_dims"]

            # Convert to a set for faster lookup
            self.angle_dims = set(self.angle_dims)
        else:
            self.angle_dims = set()

        # The dims to select for
        if("selected_dims" in loss_params):
            self.selected_dims = loss_params["selected_dims"] 
        else:
            self.selected_dims = None

    def compute_loss(self, output_dict, states):

        # Unpack the output dict to get what we need for the loss
        particles = output_dict["particles"]
        particle_weights = output_dict["particle_weights"]
        bandwidths = output_dict["bandwidths"]

        print("Sampling")
        # Draw samples
        if(bandwidths is None):
            samples = self.sample_discrete(particles, particle_weights)
        else:
            samples = self.sample_kde(particles, particle_weights, bandwidths)


        print(samples.shape)
        print("Done Sampling")

        # compute the squared errors
        all_errors = []
        for i in range(states.shape[-1]):

            # Get the state at this timestep
            state = states[...,i]

            # Only use the selected dims
            if(self.selected_dims is not None):
                if(i not in self.selected_dims):
                    continue

            # Make sure we handle angles well
            if(i in self.angle_dims):
                error = self.compute_angle_error(samples[...,i], state)
            else:
                error = self.compute_error(samples[...,i], state)

            all_errors.append(error)

        errors = torch.stack(all_errors, dim=-1)
        errors = torch.mean(errors, dim=-1)
        errors = torch.mean(errors, dim=1)

        return errors

    def sample_discrete(self, particles, particle_weights):
        
        # Need some stats
        batch_size = particles.shape[0]
        device = particles.device

        # Compute the cumulative sum of the particle weights so we can use uniform sampling
        particle_weights_cumsum = torch.cumsum(particle_weights, dim=-1)
        particle_weights_cumsum = torch.tile(particle_weights_cumsum.unsqueeze(1), [1, self.number_of_samples_to_draw, 1])

        # Generate random numbers, use the same random numbers for all batches
        uniform_random_nums = torch.rand(size=(batch_size, self.number_of_samples_to_draw, 1),device=device)

        # Select the particle indices's
        selected = particle_weights_cumsum >= uniform_random_nums
        _, selected = torch.max(selected, dim=-1)

        # Resample
        samples = torch.zeros(size=(batch_size, self.number_of_samples_to_draw, particles.shape[-1]), device=device)
        for b in range(batch_size):
            samples[b,...] = particles[b,selected[b] ,...]
        samples = torch.cat([particles[b,selected[b] ,...].unsqueeze(0) for b in range(batch_size)])

        return samples

    def sample_kde(self, particles, particle_weights, bandwidths):

        # We need the kde params for this sampling method
        if(self.kde_params is None):
            assert(False)

        # Create the KDE
        kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)

        # Sample
        return kde.sample((self.number_of_samples_to_draw,))

    def compute_error(self, samples, state):
        return (samples - state.unsqueeze(1))**2

    def compute_angle_error(self, samples, state):
        return self.compute_angle_difference(state.unsqueeze(1), samples)**2

    def compute_angle_difference(self, true_angle, predicted_angle):
        rotational_error = (predicted_angle - true_angle + np.pi) % (2 * np.pi)
        rotational_error -= np.pi 
        rotational_error[rotational_error < -np.pi] += 2.0 *np.pi
        
        return rotational_error



def create_loss_function(loss_params, model):
    assert("loss_type" in loss_params)
    loss_type = loss_params["loss_type"]

    if(loss_type == "KDE_NLL"):
        loss_function = KDE_NLL(loss_params, model)
    elif(loss_type == "MSE"):
        loss_function = MSE(loss_params, model)
    elif(loss_type == "RMSE"):
        loss_function = RMSE(loss_params, model)
    elif(loss_type == "SquaredError"):
        loss_function = SquaredError(loss_params, model)
    elif(loss_type == "AbsoluteError"):
        loss_function = AbsoluteError(loss_params, model)
    elif(loss_type == "MSESeparateAnglePosition"):
        loss_function = MSESeparateAnglePosition(loss_params, model)
    elif(loss_type == "BoundingBoxIoU"):
        loss_function = BoundingBoxIoU(loss_params, model)
    elif(loss_type == "BoundingBoxIoUSongleSolution"):
        loss_function = BoundingBoxIoUSongleSolution(loss_params, model)
    elif(loss_type == "ThesholdedDistanceError"):
        loss_function = ThesholdedDistanceError(loss_params, model)
    elif(loss_type == "ExpectedThesholdedDistanceError"):
        loss_function = ExpectedThesholdedDistanceError(loss_params, model)
    elif(loss_type == "ExpectedDistanceError"):
        loss_function = ExpectedDistanceError(loss_params, model)

    else:
        print("Unknown Loss Type {}".format(loss_type))
        assert(False)

    return loss_function
