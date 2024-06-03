import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
import pandas as pd
from matplotlib import rc,rcParams
import matplotlib.patches as mpatches


rc('font', weight='bold')

def load_data_for_experiment(experiment, number_of_runs=11, metric="nll", kernel="Gaussian"):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):

        if(kernel == "Gaussian"):

            if(metric == "nll"):
                # The file we want to load
                results_filepath = "../../../experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/table_results.pt".format(experiment, i)
            elif(metric == "rmse"):
                results_filepath = "../../../experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands_mse/table_results.pt".format(experiment, i)
            else:
                assert(False)

        elif(kernel == "Epanechnikov"):
            if(metric == "nll"):
                # The file we want to load
                results_filepath = "../../../experiments/bearings_only_experiments_methodical_epan_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/table_results.pt".format(experiment, i)
            elif(metric == "rmse"):
                results_filepath = "../../../experiments/bearings_only_experiments_methodical_epan_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands_mse/table_results.pt".format(experiment, i)
            else:
                assert(False)

            pass
        else:
            assert(False)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            # print(results_filepath)
            return None

        # Load the data 
        results = torch.load(results_filepath)  

        # Extract the NLL
        nll_mean = results[metric][0]

        # Save the data
        data.append(nll_mean)

    return data


def load_data(experiments, metric, kernel="Gaussian"):

    # All the data we loaded
    all_data = []

    # The experiments we actually loaded
    experiments_loaded = []

    # Fake the data for now
    for experiment in experiments:

        # Load the experiment data
        experiment_data = load_data_for_experiment(experiment, metric=metric, kernel=kernel)

        # Skip if we dont have data
        if(experiment_data is None):
            print(experiment, metric, kernel)
            continue

        experiment_data = [(d, experiment, kernel) for d in experiment_data]


        # Save the experiment data
        all_data.extend(experiment_data)
        experiments_loaded.append(experiment)

    # all_data = np.asarray(all_data)
    # all_data = np.transpose(all_data)

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


    experiment_names["experiment0002_implicit"] = "IRG-MDPF"
    # experiment_names["experiment0003_implicit"] = "A-MDPF-Implicit"

    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"



    experiment_names["diffy_particle_filter_learned_band"] = "TG-PF"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF"
    experiment_names["discrete_concrete"] = "C-PF"



    # experiment_names["optimal_transport_pf_learned_band_always_on_override"] = "OT-PF-AOO"
    # experiment_names["soft_resampling_particle_filter_learned_band_always_on_override"] = "SR-PF-AOO"
    # experiment_names["discrete_concrete_always_on_override"] = "C-PF-AOO"

    experiment_names["soft_resampling_particle_filter_learned_band_always_on_override"] = "SR-PF-AOO"
    experiment_names["discrete_concrete_always_on_override"] = "C-PF-AOO"


    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF-OFF"
    experiment_names["optimal_transport_pf_learned_band_always_on_override"] = "OT-PF"


    # Generate the label names in the correct order
    labels = [experiment_names[e] for e in experiments]

    return labels




def main():
    
    # # The experiments we want to load and the order we want them to be rendered in
    # experiments = []
    # experiments.append("lstm_rnn")
    # experiments.append("diffy_particle_filter_learned_band")
    # experiments.append("optimal_transport_pf_learned_band")
    # experiments.append("soft_resampling_particle_filter_learned_band")
    # experiments.append("importance_sampling_pf_learned_band")
    # experiments.append("discrete_concrete")
    # experiments.append("experiment0001")
    # experiments.append("experiment0002_implicit")
    # experiments.append("experiment0002_importance")
    # experiments.append("experiment0003_importance_init")



    experiments = []
    experiments.append("lstm_rnn")
    experiments.append("diffy_particle_filter_learned_band")
    # experiments.append("optimal_transport_pf_learned_band")
    experiments.append("optimal_transport_pf_learned_band_always_on_override")
    experiments.append("soft_resampling_particle_filter_learned_band")
    # experiments.append("soft_resampling_particle_filter_learned_band_always_on_override")
    experiments.append("importance_sampling_pf_learned_band")
    experiments.append("discrete_concrete")
    # experiments.append("discrete_concrete_always_on_override")
    experiments.append("experiment0001")
    experiments.append("experiment0002_implicit")
    experiments.append("experiment0002_importance")
    experiments.append("experiment0003_importance_init")


    metrics = []
    metrics.append(("nll", "Negative Log-likelihood"))
    metrics.append(("rmse", "Root-Mean-Square-Error"))

    for metric in metrics:

        metric_code_name = metric[0]
        metric_name = metric[1]

        # Load the data
        kernels = ["Gaussian",  "Epanechnikov"]
        all_data = []
        for kernel in kernels:
            data, experiments_loaded = load_data(experiments, metric_code_name, kernel=kernel)
            all_data.extend(data)

        df = pd.DataFrame(all_data, columns = ['value','experiment','kernel'])

        experiments_loaded = []
        for e in df["experiment"]:
            if(e not in experiments_loaded):
                experiments_loaded.append(e)

        # Plot!!!!!
        rows = 1
        cols = 1
        # fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(14, 5), squeeze=False)
        fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(14, 4.5), squeeze=False)

        # Get the axis to plot on
        ax = axes[0,0]

        ax.set_title("Bearings Only", fontsize=16, fontweight="bold")

        # Plot.  
        #   Note: set "whis" to a large number to prevent outliers from being rendered as diamonds, keep them in the min and max whiskers of the box plot
        # sns.boxplot(data=data, ax=ax, color=sns.color_palette(as_cmap=True)[0], medianprops={"color": "red"}, whis=1000000)
        sns.boxplot(data=df, x="experiment", y="value", hue="kernel", ax=ax, palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000, order=experiments_loaded)

        no_distribution_color = sns.color_palette(as_cmap=True)[2]

        if(metric_code_name == "rmse"):

            counter = 0
            to_change = [0, 1, 2, 3, 4, 5]

            for i, patch in enumerate(ax.patches): 
                if(isinstance(patch, mpatches.PathPatch) == False):
                    continue

                if(counter not in to_change):
                    continue
                counter += 1 

                # col = patch.get_facecolor()
                # patch.set_edgecolor(col)
                patch.set_facecolor(no_distribution_color)


        # Generate the labels
        labels = generate_labels(experiments_loaded)

        # # Locations of the labels
        x = np.arange(len(labels))
        ax.set_xticks(x, labels, rotation=0, fontsize=12, ha="center",va="top", weight="bold")

        # ax.set_title(metric_name, fontsize=14, weight="bold")
        ax.set_ylabel(metric_name,fontsize=14, weight="bold")

        # Format the legend
        if(metric_code_name == "nll"):
            ax.legend(fontsize=12)
        else:
            handles, labels = ax.get_legend_handles_labels()
            rect_copy = mpatches.Rectangle(handles[0].get_xy(), handles[0].get_width(), handles[0].get_height(), facecolor=no_distribution_color, edgecolor="black")
            handles.append(rect_copy)
            labels.append("No Distribution Needed")
            ax.legend(handles, labels, fontsize=12)

        # ax.legend(fontsize=12)
        ax.set(xlabel=None)


# red_patch = mpatches.Patch(color='red', label='The red data')
# plt.legend(handles=[red_patch])
    


        if(metric_code_name == "nll"):
            ax.set_ylim([5.5, 11])
        else:
            # ax.set_ylim([1.25, 4.75])
            ax.set_ylim([3, 10])
            pass

        # Make sure the labels are inside the image
        fig.tight_layout()

        # Save
        plt.savefig("bearings_only_box_plot_{}.png".format(metric_code_name))
        plt.savefig("bearings_only_box_plot_{}.pdf".format(metric_code_name))


if __name__ == '__main__':
    main()