# Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from prettytable import PrettyTable
import os
import torch
from sklearn.model_selection import ParameterGrid
import copy

# Project Imports
from evaluations.evaluation_base import *
from loss_functions.loss_functions import *



class EvaluationMetric:
    def __init__(self, metric_name, model, metric_params, save_dir):

        # Save some useful stuff
        self.metric_name = metric_name
        self.save_dir = save_dir

    def get_name(self):
        return self.metric_name

    def compute_and_add_metric(self, output_dicts, true_states, dataset_indices, dataset):
        raise NotImplemented

    def does_have_values_for_table(self):
        return True


    def get_value(self):
        raise NotImplemented

    def get_standard_deviation(self):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def save_data(self):
        raise NotImplemented

    def generate_and_save_plots(self):
        raise NotImplemented

    def get_plotting_save_dir(self):

        # Where we should save all the generated plots 
        plot_files_save_dir = "{}/plot_metric_files/".format(self.save_dir)

        # Create the raw data save directory to put all the save artifacts
        if (not os.path.exists(plot_files_save_dir)):
            os.makedirs(plot_files_save_dir)

        return plot_files_save_dir

class NonSweepEvaluationMetric(EvaluationMetric):
    def __init__(self, metric_name, model, metric_params, save_dir):
        super().__init__(metric_name, model, metric_params, save_dir)

        # Create the metric that will be used
        loss_params = metric_params["loss_params"]
        self.quantitative_metric = create_loss_function(loss_params, model)

        # All the metrics for saving
        self.all_metric_values = []

        # All the sequence level metrics and their indices
        self.sequence_level_metric_values = []
        self.aggrigated_sequence_level_metric_values = []
        self.sequence_level_indices = []

    def compute_and_add_metric(self, output_dicts, true_states, dataset_indices, dataset):

        # Compute the metric for all the steps in the sequence
        all_metric_evals = []
        for i in range(len(output_dicts)):

            output_dicts[i]["dataset_object"] = dataset

            # Compute the metric
            metric_eval = self.quantitative_metric.compute_loss(output_dicts[i], true_states[:,i,:])
            all_metric_evals.append(metric_eval)

        # Stack and reshape into the correct shape
        all_metric_evals = torch.stack(all_metric_evals).squeeze()
        all_metric_evals = torch.permute(all_metric_evals, [1,0]).squeeze()


        # Aggregate into compute the loss for this sequence
        aggregated_metric_evals = torch.mean(all_metric_evals, dim=1)


        # Save for later
        self.sequence_level_metric_values.append(all_metric_evals.cpu().detach())
        self.aggrigated_sequence_level_metric_values.append(aggregated_metric_evals.cpu().detach())
        self.sequence_level_indices.append(dataset_indices.cpu().detach())

        # Compute and save the mean over all the sequences
        # mean_aggregated_metric_evals = torch.mean(aggregated_metric_evals)
        # self.all_metric_values.append(mean_aggregated_metric_evals)


        self.all_metric_values.append(aggregated_metric_evals)


    def get_value(self):

        # Compute the final metric value
        all_metric_values_aggregated = torch.cat(self.all_metric_values)

        final_agg = self.quantitative_metric.do_final_aggrigation(all_metric_values_aggregated)

        return torch.mean(final_agg).item()

        # Compute the mean of all the metric values
        # return self.quantitative_metric.do_final_aggrigation(all_metric_values_aggregated)

    def get_standard_deviation(self):

        # Compute the final metric value
        all_metric_values_aggregated = torch.cat(self.all_metric_values)

        final_agg = self.quantitative_metric.do_final_aggrigation(all_metric_values_aggregated)

        iqr = torch.quantile(final_agg, 0.95) - torch.quantile(final_agg, 0.05)

        final_agg_list = final_agg.tolist()
        # final_agg_list.sort()
        # print(final_agg_list)
        # exit()

        # print("\n\n")
        # print("----")
        # print("max", torch.max(final_agg))
        # print("min", torch.min(final_agg))
        # print("std", torch.std(final_agg))
        # print("iqr", iqr)


        return torch.std(final_agg).item()


    def get_iqr(self):

        # Compute the final metric value
        all_metric_values_aggregated = torch.cat(self.all_metric_values)

        final_agg = self.quantitative_metric.do_final_aggrigation(all_metric_values_aggregated)

        iqr = torch.quantile(final_agg, 0.75) - torch.quantile(final_agg, 0.25)


        return iqr.item()


    # def get_standard_deviation(self):

    #     # Compute the final metric value
    #     all_metric_values_aggregated = torch.stack(self.all_metric_values)

    #     # Compute the std of all the metric values
    #     std = torch.std(all_metric_values_aggregated)


    #     return self.quantitative_metric.do_final_aggrigation(std)

    def reset(self):

        # Clear any stored values
        self.all_metric_values = []
        self.sequence_level_metric_values = []
        self.aggrigated_sequence_level_metric_values = []
        self.sequence_level_indices = []

    def save_data(self):

        # Where we should save all the raw files 
        raw_files_save_dir = "{}/raw_metric_files/".format(self.save_dir)

        # Create the raw data save directory to put all the save artifacts
        if (not os.path.exists(raw_files_save_dir)):
            os.makedirs(raw_files_save_dir)

        # Stack!!
        aggrigated_sequence_level_metric_values_stacked = torch.cat(self.aggrigated_sequence_level_metric_values)
        sequence_level_metric_values_stacked = torch.cat(self.sequence_level_metric_values)
        sequence_level_indices_stacked = torch.cat(self.sequence_level_indices)

        # Combine
        combined = torch.stack([sequence_level_indices_stacked, aggrigated_sequence_level_metric_values_stacked], dim=-1)

        # Convert the spaces in the metric name into underscores for better saving
        converted_name = self.metric_name.replace(" ", "_")

        # For the raw data we use a dict since stacking isnt favorable
        raw_data_save_dict = dict()
        raw_data_save_dict["raw_metric"] = sequence_level_metric_values_stacked
        raw_data_save_dict["indices"] = sequence_level_indices_stacked

        # Save!
        torch.save(combined, "{}/aggregated_{}.pt".format(raw_files_save_dir, converted_name))
        torch.save(raw_data_save_dict, "{}/raw_{}.pt".format(raw_files_save_dir, converted_name))

    def generate_and_save_plots(self):

        # Get where to save the plot
        plot_files_save_dir = self.get_plotting_save_dir()

        # Stack the data into a single tensor
        sequence_level_metric_values_stacked = torch.cat(self.sequence_level_metric_values)

        # Get stats
        means = torch.mean(sequence_level_metric_values_stacked, dim=0)
        stds = torch.std(sequence_level_metric_values_stacked, dim=0)

        # Generate the over time plot for this metric
        x = np.arange(0, means.shape[0])
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot()
        ax.plot(x, means, color="black", linewidth=2)
        ax.fill_between(x, means-stds, means+stds, color="red", alpha=0.5)
        ax.set_title("Error Metric \"{}\" Over Time".format(self.metric_name))
        ax.set_xlabel("Time-step Number")
        ax.set_ylabel("Metric: \"{}\"".format(self.metric_name))

        # get rid of as much white space as possible
        fig.tight_layout()

        # Convert the spaces in the metric name into underscores for better saving
        converted_name = self.metric_name.replace(" ", "_")

        # Save!
        figure_file = "{}/over_time_{}.png".format(plot_files_save_dir, converted_name)
        fig.savefig(figure_file, format="png", dpi=120, bbox_inches="tight")

        # Close to make sure things stay tidy
        plt.close("all")

# class SweepEvaluationMetric(EvaluationMetric):
#     def __init__(self, metric_name, model, metric_params, save_dir):
#         super().__init__(metric_name, model, metric_params, save_dir)

#         # Create the metric that will be used
#         loss_params = metric_params["loss_params"]

#         # Extract the sweep parameters
#         sweep_params = loss_params["sweep_params"]

#         # cant handle more than 1 right now.
#         assert(len(sweep_params) == 1) 

#         # Get all the parameters to sweep
#         all_params_to_sweep = dict()
#         for param in sweep_params:
#             sweep_name = list(param.keys())[0]

#             # Extract the info we want for sweeping
#             param_to_sweep = param[sweep_name]["param_name"]
#             starting_value = param[sweep_name]["starting_value"]
#             ending_value = param[sweep_name]["ending_value"]
#             number_of_steps = param[sweep_name]["number_of_steps"]

#             # Create the linspace for the sweep
#             sweep_values = np.linspace(starting_value, ending_value, number_of_steps)

#             # Save for later
#             all_params_to_sweep[param_to_sweep] = sweep_values

#         # Get all the parameters that we want to search over
#         self.search_parameters_grid = ParameterGrid(all_params_to_sweep)

#         # Save the model and the loss parameters
#         self.model = model
#         self.loss_params = loss_params

#         # All the batch data 
#         self.all_sequences_metric_evals = []

#     def does_have_values_for_table(self):
#         return False

#     def compute_and_add_metric(self, output_dicts, true_states, dataset_indices, dataset):

#         # Make a copy of the loss params so we can 
#         loss_params_copy = copy.deepcopy(self.loss_params)

#         all_metric_evals = []

#         # Search over all the values
#         for params in self.search_parameters_grid:

#             # Add the current params to the loss params
#             for p in params.keys():
#                 loss_params_copy[p] = params[p]


#             # Create the loss function 
#             quantitative_metric = create_loss_function(loss_params_copy, self.model)

#             all_metric_for_given_param_values = []

#             # Compute the metric for all the steps in the sequence
#             for i in range(len(output_dicts)):

#                 output_dicts[i]["dataset_object"] = dataset

#                 # Compute the metric
#                 metric_eval = quantitative_metric.compute_loss(output_dicts[i], true_states[:,i,:])
#                 all_metric_for_given_param_values.append(metric_eval)

#             # Stack and reshape into the correct shape
#             all_metric_for_given_param_values = torch.stack(all_metric_for_given_param_values).squeeze()
#             all_metric_for_given_param_values = torch.permute(all_metric_for_given_param_values, [1,0]).squeeze()


#             all_metric_evals.append(all_metric_for_given_param_values)

#         # Stack into a tensor
#         all_metric_evals = torch.stack(all_metric_evals).squeeze().cpu().detach()
    
#         # Save this batches data
#         self.all_sequences_metric_evals.append(all_metric_evals)

#     def reset(self):

#         # Clear any stored values
#         self.all_sequences_metric_evals = []

#     def save_data(self):
#         # Nothing to save
#         return

#     def generate_and_save_plots(self):

#         # Get where to save the plot
#         plot_files_save_dir = self.get_plotting_save_dir()

#         # Stack!
#         all_sequences_metric_evals_stacked = torch.cat(self.all_sequences_metric_evals, dim=1)

#         # Flatten
#         all_sequences_metric_evals_flattened = torch.reshape(all_sequences_metric_evals_stacked, [all_sequences_metric_evals_stacked.shape[0], -1])        

#         # Get stats
#         means = torch.mean(all_sequences_metric_evals_flattened, dim=1)
#         stds = torch.std(all_sequences_metric_evals_flattened, dim=1)

#         # get the x's
#         x = [] 
#         for params in self.search_parameters_grid:
#             param_name = list(params.keys())[0]
#             x.append(params[param_name])
#         x = np.asarray(x)


#         # Generate the over time plot for this metric
#         fig = plt.figure(figsize=(10, 5))
#         ax = fig.add_subplot()
#         ax.plot(x, means, color="black", linewidth=2)
#         ax.fill_between(x, means-stds, means+stds, color="red", alpha=0.5)
#         ax.set_title("Error Metric \"{}\" Over Time".format(self.metric_name))
#         ax.set_xlabel("Time-step Number")
#         ax.set_ylabel("Metric: \"{}\"".format(self.metric_name))

#         # get rid of as much white space as possible
#         fig.tight_layout()

#         # Convert the spaces in the metric name into underscores for better saving
#         converted_name = self.metric_name.replace(" ", "_")

#         # Save!
#         figure_file = "{}/sweeped_{}.png".format(plot_files_save_dir, converted_name)
#         fig.savefig(figure_file, format="png", dpi=120, bbox_inches="tight")

#         # Close to make sure things stay tidy
#         plt.close("all")





# class MultipleTrialsEvaluationMetric(EvaluationMetric):
#     def __init__(self, metric_name, model, metric_params, save_dir):
#         super().__init__(metric_name, model, metric_params, save_dir)

#         # Create the metric that will be used
#         loss_params = metric_params["loss_params"]
#         self.quantitative_metric = create_loss_function(loss_params, model)

#         # All the metrics for saving
#         self.all_metric_values = []

#     def compute_and_add_metric(self, output_dicts, true_states, dataset_indices, dataset):

#         # Compute the metric for all the steps in the sequence
#         all_metric_evals = []
#         for i in range(len(output_dicts)):

#             output_dicts[i]["dataset_object"] = dataset

#             # Compute the metric
#             metric_eval = self.quantitative_metric.compute_loss(output_dicts[i], true_states[:,i,:])
#             all_metric_evals.append(metric_eval)

#         # Stack and reshape into the correct shape
#         all_metric_evals = torch.stack(all_metric_evals).squeeze()
#         all_metric_evals = torch.permute(all_metric_evals, [1,0]).squeeze()

#         # Aggregate into compute the loss for this sequence
#         aggregated_metric_evals = torch.mean(all_metric_evals, dim=1)

#         # Compute and save the mean over all the sequences
#         mean_aggregated_metric_evals = torch.mean(aggregated_metric_evals)
#         self.all_metric_values.append(mean_aggregated_metric_evals)

#     def get_value(self):

#         # Compute the final metric value
#         all_metric_values_aggregated = torch.stack(self.all_metric_values)

#         # Compute the mean of all the metric values
#         return self.quantitative_metric.do_final_aggrigation(all_metric_values_aggregated)

#     def get_standard_deviation(self):

#         # Compute the final metric value
#         all_metric_values_aggregated = torch.stack(self.all_metric_values)

#         # Compute the std of all the metric values
#         std = torch.std(all_metric_values_aggregated)


#         return self.quantitative_metric.do_final_aggrigation(std)

#     def reset(self):

#         # Clear any stored values
#         self.all_metric_values = []



class FullSequenceEvaluationBase(EvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, save_dir, device, seed)

        #Save the model
        self.model = model

        # Parse the evaluation parameters
        evaluation_params = experiment["evaluation_params"]
        self.number_to_render = evaluation_params["number_to_render"]
        self.number_of_particles = evaluation_params["number_of_particles"]
        self.render_particles = evaluation_params["render_particles"]

        # Check if we should do the qualitative
        if("do_qualitative" in evaluation_params):
            self.do_qualitative = evaluation_params["do_qualitative"]

        else:
            self.do_qualitative = True

        # See if we have specific sequences we should render, otherwise move on
        if("sequences_to_render" in evaluation_params):
            self.sequences_to_render = evaluation_params["sequences_to_render"]
        else:
            self.sequences_to_render = None

        # Parse the quantitative evaluation params if we have them
        if("quantitative_evaluation_params" in evaluation_params):

            # Parse the parameters
            quantitative_evaluation_params = evaluation_params["quantitative_evaluation_params"]
            self.do_quantitative = quantitative_evaluation_params["do_quantitative"]
            self.quantitative_batch_size = quantitative_evaluation_params["batch_size"]


            # Check if we should do the qualitative
            if("save_raw_sequence_outputs" in quantitative_evaluation_params):
                self.save_raw_sequence_outputs = quantitative_evaluation_params["save_raw_sequence_outputs"]

            else:
                self.save_raw_sequence_outputs = False


            self.metrics = []
            self.multiple_trial_metric = []

            # Create all the metrics
            for metric_name in quantitative_evaluation_params["metrics"]:

                # Extract the specific params for this metric
                metric_params = quantitative_evaluation_params["metrics"][metric_name]

                # Check if this is a sweep metric or not
                if("sweep_params" in metric_params["loss_params"]):
                    # Create the metric
                    # self.metrics.append(SweepEvaluationMetric(metric_name, self.model, metric_params, self.save_dir))
                    assert(False)

                else:
                    # Create the metric
                    self.metrics.append(NonSweepEvaluationMetric(metric_name, self.model, metric_params, self.save_dir))
                    # self.multiple_trial_metric.append(MultipleTrialsEvaluationMetric(metric_name, self.model, metric_params, self.save_dir))

        else:

            # No params so we dont do quantitative 
            self.do_quantitative = False

        # Parse the quantitative evaluation params if we have them
        if("render_panel_params" in evaluation_params):

            # Parse the parameters
            render_panel_params = evaluation_params["render_panel_params"]
            self.do_render_panel = render_panel_params["do_render_panel"]
            self.render_panel_modulo = render_panel_params["render_panel_modulo"]
            self.render_panel_num_cols = render_panel_params["render_panel_num_cols"]
            self.render_panel_num_rows = render_panel_params["render_panel_num_rows"]
            self.render_panel_must_include_indices = render_panel_params["render_panel_must_include_indices"]

        else:

            # No params so we dont do quantitative 
            self.do_render_panel = False


        # If we should us a manually set bandwidth if we need a bandwidth
        if("use_manual_bandwidth" in evaluation_params):

            self.use_manual_bandwidth = evaluation_params["use_manual_bandwidth"]

            # If we are going to use a bandwidth then we should make sure we have one
            if(self.use_manual_bandwidth):
                assert("manual_bandwidth" in evaluation_params)
                self.manual_bandwidth = evaluation_params["manual_bandwidth"]
            else:
                self.manual_bandwidth = None

        else:
            self.use_manual_bandwidth = False


        # If the model outputs a KDE then lets get a KDE going
        if(self.model.outputs_kde()):   
            self.kde_params = evaluation_params["kde_params"]
        
        elif(self.use_manual_bandwidth and (self.manual_bandwidth is not None)):
            # In this case we may or may not need the params.  So try to load them and if they are not there
            # and we need them later, we should fail later.  This is ugly but whatever in the name of 
            # flexibility and what not....
            if("kde_params" in evaluation_params):
                self.kde_params = evaluation_params["kde_params"]
            else:
                self.kde_params = None

        else:
            # No KDE params are needed here (hopefully.)
            self.kde_params = None

    def get_rows_and_cols(self):
        raise NotImplemented

    def render_frame(self, frame_number, data, output_dicts, axes):
        raise NotImplemented        

    def run_evaluation(self):

        # Set the model to be eval mode for evaluation
        self.model.eval()

        # If we are to do the full metric evaluation then run it!
        if(self.do_quantitative):
            self.evaluate_quantatitively()

        # Do some renderings
        if(self.do_qualitative or self.do_render_panel):
            self.evaluate_qualitively()

    def evaluate_qualitively(self):

        # Create a list of shuffled indices for the dataset
        # dataset_values = list(range(len(self.evaluation_dataset)))
        # random.shuffle(dataset_values)

        if(self.sequences_to_render is not None):
            dataset_to_use = torch.utils.data.Subset(self.evaluation_dataset, self.sequences_to_render)
            num_to_render = min(len(self.sequences_to_render), self.number_to_render)
        else:
            dataset_to_use = self.evaluation_dataset
            num_to_render = self.number_to_render

        # Create a data-loader to load data from
        quantitative_data_loader = torch.utils.data.DataLoader(dataset_to_use, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=self.evaluation_dataset.get_collate_fn())
        quantitative_data_loader_iter = iter(quantitative_data_loader)

        print("Rendering Sequences")

        # Render each experiment
        for i in tqdm(range(num_to_render)):

            # Get a sample of data
            # data = self.evaluation_dataset[dataset_values[i]]
            data = next(quantitative_data_loader_iter)

            # run the model
            output_dicts = self.run_the_model(data)

            # Aggregate the outputs from a list of dicts to a dict of aggregated data
            output_dicts = self.aggregate_output_dicts(output_dicts)

            # Render the sequence
            if(self.do_qualitative):
                self.render_sequence(i, data, output_dicts)

            # Render the sequence
            if(self.do_render_panel):
                self.render_panel(i, data, output_dicts)

    def evaluate_quantatitively(self):

        # Lets be speedy and not use gradients here!
        with torch.no_grad():

            # Create a dataloader so we can batch!
            data_loader = torch.utils.data.DataLoader(dataset=self.evaluation_dataset, batch_size=self.quantitative_batch_size, shuffle=False, num_workers=6, pin_memory=True, collate_fn=self.evaluation_dataset.get_collate_fn())

            # for trial_idx in tqdm(range(10), leave=False):
            for trial_idx in tqdm(range(1), leave=False):

                
                all_sequence_data = dict()

                # Go through all the data, running the model and computing the metric scores
                t = tqdm(iter(data_loader), leave=False, total=len(data_loader))
                for step, data in enumerate(t):
                        
                    # Get the dataset indices
                    dataset_indices = data["dataset_index"]

                    # run the model
                    output_dicts = self.run_the_model(data)

                    if(self.save_raw_sequence_outputs):
                        agg_output_dicts = self.aggregate_output_dicts(output_dicts, keep_all=False)
                        for i in range(dataset_indices.shape[0]):

                            selected_dict = dict()
                            for k in agg_output_dicts.keys():
                                selected_dict[k] = agg_output_dicts[k][:, i, ...]
                            all_sequence_data[dataset_indices[i].item()] = selected_dict


                    # Extract the true states
                    states = data["states"].to(self.device)

                    # Compute the metrics for only the first run
                    if(trial_idx == 0):
                        # Compute all the metrics
                        for metric in self.metrics:
                            metric.compute_and_add_metric(output_dicts, states, dataset_indices, self.evaluation_dataset)

                    # for metric in self.multiple_trial_metric:
                        # metric.compute_and_add_metric(output_dicts, states, dataset_indices, self.evaluation_dataset)




            ################################################################################################
            ################################################################################################
            # Print and save the data
            ################################################################################################
            ################################################################################################

            if(self.save_raw_sequence_outputs):
                torch.save(all_sequence_data, "{}/raw_all_sequence_evaluation_run_data.pt".format(self.save_dir))

            # Do the final saving/plotting/whatever
            for metric in self.metrics:
                metric.save_data()
                metric.generate_and_save_plots()

            # Write a Table with the results
            results_table = PrettyTable()
            results_table.field_names = ["Metric", "Value", "Standard Deviation"]
            for metric in self.metrics:

                # Skip metrics that dont have values that can be inserted into the table
                if(metric.does_have_values_for_table() == False):
                    continue

                results_table.add_row([metric.get_name(),"{:0.4f}".format(metric.get_value()), "{:0.4f}".format(metric.get_standard_deviation())])

            # Print the table
            print(results_table)

            # Save the table
            output_file = open(self.save_dir + "/output.txt", "a")
            output_file.write("============================================================================\n")
            output_file.write("Full Sequence Evaluation\n")
            output_file.write("Hyper-parameters:\n")
            output_file.write("\t - Sequence Length: {}\n".format(self.evaluation_dataset.get_subsequence_length()))
            output_file.write("\t - Number of Particles: {}\n".format(self.number_of_particles))
            output_file.write("\n")
            output_file.write(str(results_table))
            output_file.write("\n")
            output_file.write("============================================================================\n")
            output_file.close()

            # Save the results to a file so we can aggregate them later
            table_results = dict()
            for metric in self.metrics:
            # for metric in self.multiple_trial_metric:

                # Skip metrics that dont have values that can be inserted into the table
                if(metric.does_have_values_for_table() == False):
                    continue

                table_results[metric.get_name()] = (metric.get_value(), metric.get_standard_deviation(), metric.get_iqr())

            torch.save(table_results, "{}/table_results.pt".format(self.save_dir))



    def run_the_model(self, data):

        # We dont want the gradient when we run the model
        with torch.no_grad():

            # Get the sub-sequence length
            subsequence_length = self.evaluation_dataset.get_subsequence_length()

            # Unpack the data and move to the device
            observations = data["observations"].to(self.device)
            states = data["states"]#.to(device)

            # Add a batch dimentions to make everything work if one is not present
            if(len(states.shape) == 2):
                observations = observations.unsqueeze(0)
                states = states.unsqueeze(0)                

            # if we have an action set then set the action
            if("actions" in data):
                actions = data["actions"]
                if(actions is not None):
                    actions = actions.to(self.device)

                    # Add a batch dimentions to make everything work if one is not present
                    if(len(actions.shape) == 2):  
                        actions = actions.unsqueeze(0)
            else:
                actions = None

            # Sometimes we dont have reference patches
            if("reference_patch" in data):
                reference_patch = data["reference_patch"]
                if(reference_patch is not None):
                    reference_patch = reference_patch.to(self.device, non_blocking=True)
            else:
                reference_patch = None

            # Sometimes we dont have reference patches
            if("world_map" in data):
                world_map = data["world_map"]
                if(world_map is not None):
                    # world_map = [w.to(self.device, non_blocking=True) for w in world_map]
                    world_map = world_map.to(self.device, non_blocking=True)
            else:
                world_map = None



            # transform the observation
            transformed_observation = self.problem.observation_transformer.forward_tranform(observations)

            # Create the initial state for the particle filter
            output_dict = self.model.create_initial_dpf_state(states[:,0,:],transformed_observation, self.number_of_particles)

            # We want to keep track of all the output dicts so we can use that to render things later
            all_output_dicts = []
            all_output_dicts.append(output_dict)

            # Run through the sub-sequence
            for seq_idx in tqdm(range(subsequence_length-1), leave=False, desc="Running Model"):

                # Get the observation for this step
                observation = transformed_observation[:,seq_idx+1,:]


                # Get the observation for the next step (if there is a next step)
                if((seq_idx+2) >= subsequence_length):
                    next_observation = None
                else:
                    next_observation = transformed_observation[:,seq_idx+2,:]

                # Create the next input dict from the last output dict
                input_dict = dict()
                for key in output_dict.keys():

                    if(isinstance(output_dict[key], torch.Tensor)):
                        input_dict[key] = output_dict[key].clone()
                    else:
                        input_dict[key] = output_dict[key]

                input_dict["observation"] = observation
                input_dict["next_observation"] = next_observation
                input_dict["reference_patch"] = reference_patch
                input_dict["world_map"] = world_map


                if(actions is not None):
                    input_dict["actions"] = actions[:,seq_idx, :]
                else:
                    input_dict["actions"] = None

                # We have no timestep information
                input_dict["timestep_number"] = None

                # Run the model on this step
                output_dict = self.model(input_dict)
                all_output_dicts.append(output_dict)

            # Return the output dict 
            return all_output_dicts

    def aggregate_output_dicts(self, all_output_dicts, keep_all=False):

        # get all the keys
        dict_keys = set()
        for output_dict in all_output_dicts:            
            for key in output_dict.keys():
                dict_keys.add(key)

        if(keep_all == False):
            # Remove keys if they are present
            keys_to_remove = ["timestep_number", "observation", "actions", "next_observation"]
            for key in keys_to_remove:
                if(key in dict_keys):
                    dict_keys.remove(key)

        # Convert from set to list to make it easy to use
        dict_keys = list(dict_keys)

        # Aggregate!! 
        aggregated_output_dicts = dict()
        for key in dict_keys:
            all_data_for_key = []

            # Get the type of the data.  
            # Since we can only aggregate tensors, ignore non-tensors
            key_sample_data = all_output_dicts[0][key]
            if(isinstance(key_sample_data, torch.Tensor) == False):
                continue

            # Aggregate the data
            for output_dict in all_output_dicts:
                data = output_dict[key]
                all_data_for_key.append(data)

            # Stack them into a single tensor!!
            all_data_for_key = torch.stack(all_data_for_key)
            
            # Save it based on the key
            aggregated_output_dicts[key] = all_data_for_key

        return aggregated_output_dicts

    def add_ess_to_plot(self, ax, weights, weights_for):

        # Get some stats
        sequence_length = weights.shape[0]
        number_of_particles = weights.shape[2]

        # Compute the effective sample size
        ess = weights.squeeze(1)**2
        ess = torch.sum(ess, dim=-1)
        ess = 1.0 / ess

        # print(ess)

        # exit()

        # Plot the ess
        ax.plot(ess.cpu().numpy(), label="Effective Sample Size", color="black")
        ax.set_title("Effective Sample Size State (N={}) for {}".format(number_of_particles, weights_for))
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Effective Sample Size")
        ax.set_xlim([0, sequence_length])

        ess_list = ess.cpu().squeeze().tolist()

        # Add labels every so often so we can read the data
        for i in range(0, len(ess_list), 5):
            ax.text(i, ess_list[i], "{:.2f}".format(ess_list[i]))


