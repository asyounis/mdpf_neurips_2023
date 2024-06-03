import numpy as np
import os
import torch
from prettytable import PrettyTable




def load_data_for_experiment(experiment, number_of_runs=11, metric="nll"):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):

        if(metric == "nll"):
            # The file we want to load
            results_filepath = "../../../experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/table_results.pt".format(experiment, i)
        elif(metric == "rmse"):
            results_filepath = "../../../experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands_mse/table_results.pt".format(experiment, i)
        else:
            assert(False)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            return None

        # Load the data 
        results = torch.load(results_filepath)  

        # Extract the NLL
        nll_mean = results[metric][0]

        # Save the data
        data.append(nll_mean)

    return data

def load_data(experiments, metric):

    # All the data we loaded
    all_data = []

    # The experiments we actually loaded
    experiments_loaded = []

    # Fake the data for now
    for experiment in experiments:

        # Load the experiment data
        experiment_data = load_data_for_experiment(experiment, metric=metric)

        # Skip if we dont have data
        if(experiment_data is None):
            print(experiment, metric)
            continue

        # Save the experiment data
        all_data.append(experiment_data)
        experiments_loaded.append(experiment)

    all_data = np.asarray(all_data)
    all_data = np.transpose(all_data)

    return all_data, experiments_loaded



def generate_labels(experiments):

    # The names we want to give to the experiments
    experiment_names = dict()
    experiment_names["lstm_rnn"] = "LSTM"
    # experiment_names["diffy_particle_filter"] = "TG-PF"
    # experiment_names["optimal_transport_pf"] = "OT-PF"
    # experiment_names["soft_resampling_particle_filter"] = "SR-PF"
    # experiment_names["importance_sampling_pf"] = "DIS-PF"

    experiment_names["experiment0001"] = "TG-MDPF"
    experiment_names["experiment0002_importance"] = "MDPF"
    # experiment_names["experiment0003_importance"] = "A-MDPF"
    # experiment_names["experiment0003_importance_init"] = "A-MDPF-Init"
    experiment_names["experiment0003_importance_init"] = "A-MDPF"


    experiment_names["experiment0002_implicit"] = "MDPF-Implicit"
    experiment_names["experiment0003_implicit"] = "A-MDPF-Implicit"

    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"



    experiment_names["diffy_particle_filter_learned_band"] = "TG-PF"
    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF"

    # Generate the label names in the correct order
    labels = [experiment_names[e] for e in experiments]

    return labels



def main():
	
    # The experiments we want to load and the order we want them to be rendered in
    experiments = []
    # experiments.append("diffy_particle_filter")
    # experiments.append("optimal_transport_pf")
    # experiments.append("soft_resampling_particle_filter")
    # experiments.append("importance_sampling_pf")
    experiments.append("diffy_particle_filter_learned_band")
    experiments.append("optimal_transport_pf_learned_band")
    experiments.append("soft_resampling_particle_filter_learned_band")
    experiments.append("importance_sampling_pf_learned_band")
    experiments.append("experiment0002_concrete")
    experiments.append("experiment0003_concrete")
    experiments.append("experiment0001")
    experiments.append("experiment0002_importance")
    experiments.append("experiment0002_implicit")
    # experiments.append("experiment0003_importance")
    experiments.append("experiment0003_importance_init")


    metrics = []
    metrics.append(("nll", "Negative Log-likelihood"))
    metrics.append(("rmse", "Root-Mean-Square-Error"))

    for metric in metrics:

        metric_code_name = metric[0]
        metric_name = metric[1]

        # Load the data
        data, experiments_loaded = load_data(experiments, metric_code_name)

        # Make the table
        table = PrettyTable()

        # Add the names to the header 
        field_names = generate_labels(experiments_loaded)
        field_names.insert(0, "Run Number")
        table.field_names = field_names

        # Fill in the rows
        for i in range(data.shape[0]):

            # Limit to 3 decimal places
            row = ["{:0.3f}".format(v) for v in data[i].tolist()]
            row.insert(0, i)

            # Add it to the table
            table.add_row(row)


        # Show!!!
        print(metric_name)
        print(table)
        print("")
        print("")
        print("")
        print("")







if __name__ == '__main__':
	main()