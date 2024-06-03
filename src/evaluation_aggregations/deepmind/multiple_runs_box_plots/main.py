import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
import pandas as pd
from matplotlib import rc,rcParams


rc('font', weight='bold')



def load_data_for_experiment(experiment, maze_number, number_of_runs=5, metric="nll", kernel="Normal"):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):

        if(kernel == "Normal"):
            if(metric == "nll"):
                # The file we want to load
                results_filepath = "../../../experiments/deepmind_maze_experiments_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}/table_results.pt".format(experiment, i, maze_number)
            # elif(metric == "rmse"):
            elif("rmse" in metric):
                results_filepath = "../../../experiments/deepmind_maze_experiments_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}_mse/table_results.pt".format(experiment, i, maze_number)
            else:
                assert(False)

        elif(kernel == "Epanechnikov"):
            assert(False)

            if(metric == "nll"):
                # The file we want to load
                results_filepath = "../../../experiments/deepmind_maze_experiments_epan_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}/table_results.pt".format(experiment, i, maze_number)
            elif(metric == "rmse"):
                if("experiment" in experiment):
                    results_filepath = "../../../experiments/deepmind_maze_experiments_epan_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}_mse/table_results.pt".format(experiment, i, maze_number)
                else:
                    results_filepath = "../../../experiments/deepmind_maze_experiments_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}_mse/table_results.pt".format(experiment, i, maze_number)
            else:
                assert(False)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            print(results_filepath)
            return None

        # Load the data 
        results = torch.load(results_filepath)  

        # Extract the NLL
        nll_mean = results[metric][0]

        # Save the data
        data.append(nll_mean)

    return data


def load_data(experiments, maze_number, metric, kernel):

    # All the data we loaded
    all_data = []

    # The experiments we actually loaded
    experiments_loaded = []

    # Fake the data for now
    for experiment in experiments:

        # Load the experiment data
        # if(maze_number > 1):
        #     experiment_data = load_data_for_experiment(experiment, 1, metric=metric)
        # else:
        #     experiment_data = load_data_for_experiment(experiment, maze_number, metric=metric)

        experiment_data = load_data_for_experiment(experiment, maze_number, metric=metric,kernel=kernel)

        # Skip if we dont have data
        if(experiment_data is None):
            print(experiment, metric)
            continue



        experiment_data = [(d, experiment, "Maze {}".format(maze_number)) for d in experiment_data]

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
    experiment_names["experiment0003_importance"] = "A-MDPF-BOOOOOO"
    # experiment_names["experiment0003_importance_init"] = "A-MDPF-Init"


    experiment_names["experiment0003_importance_init"] = "A-MDPF"
    experiment_names["experiment0003_importance_init_dis"] = "A-MDPF"
    experiment_names["experiment0003_importance_init_dis_local"] = "A-MDPF"

    experiment_names["experiment0002_implicit"] = "IRG-MDPF"



    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"

    experiment_names["discrete_concrete"] = "C-PF"


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
    # experiments.append("lstm_rnn")
    # experiments.append("diffy_particle_filter_learned_band")
    # experiments.append("optimal_transport_pf_learned_band")
    # experiments.append("soft_resampling_particle_filter_learned_band")
    # experiments.append("importance_sampling_pf_learned_band")
    # experiments.append("discrete_concrete")
    # experiments.append("experiment0001")
    # experiments.append("experiment0002_implicit")
    # experiments.append("experiment0002_importance")
    # # experiments.append("experiment0003_importance_init")

    # experiments.append("experiment0003_importance_init")
    # # experiments.append("experiment0003_importance_init_dis")
    # # experiments.append("experiment0003_importance_init_dis_local")




    experiments.append("lstm_rnn")
    experiments.append("diffy_particle_filter_learned_band")
    experiments.append("optimal_transport_pf_learned_band")
    experiments.append("soft_resampling_particle_filter_learned_band")
    experiments.append("importance_sampling_pf_learned_band")
    experiments.append("discrete_concrete")
    experiments.append("experiment0001")
    experiments.append("experiment0002_implicit")
    experiments.append("experiment0002_importance")
    experiments.append("experiment0003_importance_init")



    metrics = []
    metrics.append(("nll", "Negative Log-likelihood"))
    metrics.append(("rmse", "Root-Mean-Square-Error"))
    # metrics.append(("rmse_position", "RMSE-Pos"))
    # metrics.append(("rmse_angle", "RMSE-Angle"))


    # kernels = ["Normal", "Epanechnikov"]
    kernels = ["Normal"]
    for kernel in kernels:



        maze_numbers = [1, 2, 3]
        for metric in metrics:

            all_data = []
            
            metric_code_name = metric[0]
            metric_name = metric[1]


            for maze_number in maze_numbers:
                # Load the data
                data, experiments_loaded = load_data(experiments, maze_number, metric_code_name, kernel)
                # all_data.append(data)

                all_data.extend(data)

            df = pd.DataFrame(all_data, columns = ['value','experiment','maze_number'])

            experiments_loaded = []
            for e in df["experiment"]:
                if(e not in experiments_loaded):
                    experiments_loaded.append(e)

            data = all_data

            # Plot!!!!!
            rows = 1
            cols = 1
            # fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 4), squeeze=False)
            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 3.75), squeeze=False)

            # Get the axis to plot on
            ax = axes[0,0]

            ax.set_title("Deepmind Maze", fontsize=16, fontweight="bold")

            # Plot.  
            #     Note: set "whis" to a large number to prevent outliers from being rendered as diamonds, keep them in the min and max whiskers of the box plot
            # sns.boxplot(data=data,x=["k"], ax=ax, color=sns.color_palette(as_cmap=True)[3], medianprops={"color": "red"}, boxprops={"facecolor": (.4, .6, .8, .5)}, whis=1000000)
            # sns.boxplot(data=df, x="experiment", y="value", hue="maze_number", ax=ax, color=sns.color_palette(as_cmap=True)[3], medianprops={"color": "red"}, boxprops={"facecolor": (.4, .6, .8, .5)}, orient="v", whis=1000000)
            sns.boxplot(data=df, x="experiment", y="value", hue="maze_number", ax=ax, palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000)

            # Generate the labels
            labels = generate_labels(experiments_loaded)

            # Locations of the labels
            x = np.arange(len(labels))
            ax.set_xticks(x, labels, rotation=0, fontsize=12, ha="center",va="top", weight="bold")

            # ax.set_title(metric_name, fontsize=14, weight="bold")
            # ax.set_xlabel("Method",fontsize=14, weight="bold")
            ax.set_ylabel(metric_name,fontsize=14, weight="bold")

            # Format the legend
            ax.legend(fontsize=12)
            ax.set(xlabel=None)


            if(metric_code_name == "nll" and kernel == "Normal"):
                # Draw the observation
                inset_ax = ax.inset_axes([0.01,0.69,0.06,0.3])
                # inset_ax.set_yticklabels([])
                # inset_ax.set_xticklabels([])
                # inset_ax.tick_params(left=False, bottom=False)
                inset_ax.patch.set_edgecolor('black')  
                inset_ax.patch.set_linewidth(1.5)  
                tmp = df[df["experiment"] == "lstm_rnn"]
                sns.boxplot(data=tmp, x="experiment", y="value", hue="maze_number", ax=inset_ax, palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000)
                inset_ax.legend().remove()
                inset_ax.set(xlabel=None)
                inset_ax.set(ylabel=None)
                inset_ax.yaxis.set_label_position("right")
                inset_ax.yaxis.tick_right()


                labels = ["LSTM"]
                x = np.arange(len(labels))
                inset_ax.set_xticks(x, labels, rotation=0, fontsize=9, ha="center",va="top", weight="bold")


               # Draw the observation
                inset_ax = ax.inset_axes([0.72,0.69,0.06,0.3])
                # inset_ax.set_yticklabels([])
                # inset_ax.set_xticklabels([])
                # inset_ax.tick_params(left=False, bottom=False)
                inset_ax.patch.set_edgecolor('black')  
                inset_ax.patch.set_linewidth(1.5)  
                tmp = df[df["experiment"] == "experiment0002_implicit"]
                sns.boxplot(data=tmp, x="experiment", y="value", hue="maze_number", ax=inset_ax, palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000)
                inset_ax.legend().remove()
                inset_ax.set(xlabel=None)
                inset_ax.set(ylabel=None)
                inset_ax.yaxis.set_label_position("right")
                inset_ax.yaxis.tick_right()


                labels = ["IRG-MDPF"]
                x = np.arange(len(labels))
                inset_ax.set_xticks(x, labels, rotation=0, fontsize=9, ha="center",va="top", weight="bold")


            if("rmse" in metric_code_name and kernel == "Normal"):



               # Draw the observation
                inset_ax = ax.inset_axes([0.72,0.69,0.06,0.3])
                # inset_ax.set_yticklabels([])
                # inset_ax.set_xticklabels([])
                # inset_ax.tick_params(left=False, bottom=False)
                inset_ax.patch.set_edgecolor('black')  
                inset_ax.patch.set_linewidth(1.5)  
                tmp = df[df["experiment"] == "experiment0002_implicit"]
                sns.boxplot(data=tmp, x="experiment", y="value", hue="maze_number", ax=inset_ax, palette=sns.color_palette(as_cmap=True), medianprops={"color": "red"}, orient="v", whis=1000000)
                inset_ax.legend().remove()
                inset_ax.set(xlabel=None)
                inset_ax.set(ylabel=None)
                inset_ax.yaxis.set_label_position("right")
                inset_ax.yaxis.tick_right()


                labels = ["IRG-MDPF"]
                x = np.arange(len(labels))
                inset_ax.set_xticks(x, labels, rotation=0, fontsize=9, ha="center",va="top", weight="bold")





            if(kernel == "Normal"):
                if(metric_code_name == "nll"):
                    ax.set_ylim([5.5, 10])
                elif(metric_code_name == "rmse_angle"):
                    # ax.set_ylim([0.5, 2])
                    pass
                elif(metric_code_name == "rmse_position"):
                    ax.set_ylim([3.0, 5.5])
                    # ax.set_ylim([2, 4.0])
                    pass

                else:
                    # ax.set_ylim([1.25, 2.8])
                    ax.set_ylim([3.25, 5.75])
                    # ax.set_ylim([120, 450])
                    pass

            # Make sure the labels are inside the image
            fig.tight_layout()

            # Save
            plt.savefig("{}_box_plot_{}.png".format(kernel, metric_code_name))
            plt.savefig("{}_box_plot_{}.pdf".format(kernel, metric_code_name))




if __name__ == '__main__':
    main()