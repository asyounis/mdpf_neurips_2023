
# Imports
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import os
import numpy as np


def load_and_preprocess_file(filepath):

    # Load the data
    loaded_data = torch.load(filepath)

    # a = torch.randperm(1000).unsqueeze(1)
    # b = torch.randperm(1000).unsqueeze(1)
    # loaded_data = torch.cat([a,b], dim=-1)

    # Sort by ID
    # sorted_data, inds = torch.sort(loaded_data, stable=True, dim=0)
    _, inds = torch.sort(loaded_data, stable=True, dim=0)
    sorted_data = loaded_data[inds[:, 0]]

    # print(sorted_data.shape)

    # for i in range(sorted_data.shape[0]):
    #     # print(sorted_data[i])

    #     a = sorted_data[i, 0]
    #     b = sorted_data[i, 1]

    #     print(a, b, loaded_data[a])

        # assert(b == loaded_data[a, 1])


    # for i in range(loaded_data.shape[0]):
    #     # print(loaded_data[i])

    #     a = int(loaded_data[i, 0])
    #     b = loaded_data[i, 1]

    #     print(a, b, sorted_data[a])

    #     assert(b == sorted_data[a, 1])


    # exit()
    # print(loaded_data[999])

    # for i in range(loaded_data.shape[0]):
        # print(loaded_data[i])

    # exit()

    return sorted_data
    
def find_min_and_max_values(data_a, data_b):

    # Combine so we can get the min and max values
    combined = torch.cat([data_a[:,1], data_b[:,1]])

    # Get the min and max values
    min_value = torch.min(combined).item()
    max_value = torch.max(combined).item()

    # Get the quantile max
    max_quantile = torch.quantile(combined, 0.9, interpolation="nearest")

    return min_value, max_value, max_quantile

def compare(data_a_name, data_b_name, maze_number, metric, title, save_file_name):

    # Load the data
    # data_a = load_and_preprocess_file("../../../experiments/deepmind_maze_experiments_no_actions/{}/saves/full_dpf_evaluation_maze_{}/raw_metric_files/aggregated_{}.pt".format(data_a_name, maze_number, metric))
    # data_b = load_and_preprocess_file("../../../experiments/deepmind_maze_experiments_no_actions/{}/saves/full_dpf_evaluation_maze_{}/raw_metric_files/aggregated_{}.pt".format(data_b_name, maze_number, metric))

    data_a = load_and_preprocess_file("../../../experiments/deepmind_maze_experiments/{}/saves/full_dpf_evaluation_maze_{}/raw_metric_files/aggregated_{}.pt".format(data_a_name, maze_number, metric))
    data_b = load_and_preprocess_file("../../../experiments/deepmind_maze_experiments/{}/saves/full_dpf_evaluation_maze_{}/raw_metric_files/aggregated_{}.pt".format(data_b_name, maze_number, metric))

    # Compute on which side of the line it is
    data_a_bigger_than_b = torch.sum(data_a[:, 1] > data_b[:, 1]).item()
    data_b_bigger_than_a = torch.sum(data_b[:, 1] > data_a[:, 1]).item()

    # Perform the Wilcoxon signed-rank test
    wilcoxon_result = wilcoxon(data_a[:, 1].numpy(), data_b[:, 1].numpy())

    # Get the min and max values for plotting 
    min_value, max_value, max_quantile = find_min_and_max_values(data_a, data_b)

    rows = 1
    cols = 1
    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(6*cols, 6*rows), squeeze=False)

    #  The full range plot
    axes[0, 0].scatter(data_a[:,1].numpy(), data_b[:,1].numpy(), color="black", s=1)
    axes[0, 0].plot([min_value, max_value], [min_value, max_value], color="red")
    axes[0, 0].set_xlabel(data_a_name)
    axes[0, 0].set_ylabel(data_b_name)
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlim([min_value, max_value])
    axes[0, 0].set_ylim([min_value, max_value])

    # Add text to the plot
    text_str = "Count: {}".format(data_a_bigger_than_b)
    axes[0, 0].text(0.9, 0.75, text_str, horizontalalignment='center', verticalalignment='center',transform=axes[0, 0].transAxes, weight="bold")

    text_str = "Count: {}".format(data_b_bigger_than_a)
    axes[0, 0].text(0.75, 0.9, text_str, horizontalalignment='center', verticalalignment='center',transform=axes[0, 0].transAxes, weight="bold")

    text_str = "Wilcoxon Test\npvalue: {:0.4f}     statistic: {:05d}".format(wilcoxon_result.pvalue, int(wilcoxon_result.statistic))
    axes[0, 0].text(0.7, 0.05, text_str, horizontalalignment='center', verticalalignment='center',transform=axes[0, 0].transAxes, color="red", weight="bold")



    # #  The scaled plot so we can see the data better
    # axes[0, 1].scatter(data_a[:,1].numpy(), data_b[:,1].numpy(), color="black", s=1)
    # axes[0, 1].plot([min_value, max_quantile], [min_value, max_quantile], color="red")
    # axes[0, 1].set_xlabel(data_a_name)
    # axes[0, 1].set_ylabel(data_b_name)
    # axes[0, 1].set_title(title)
    # axes[0, 1].set_xlim([min_value, max_quantile])
    # axes[0, 1].set_ylim([min_value, max_quantile])

    fig.tight_layout(h_pad=1, w_pad=1)

    plt.savefig(save_file_name)
    plt.close('all')

    # return the P value so we can plot it in a matrix
    return wilcoxon_result.pvalue

def plot_p_values(all_p_values, experiments, maze_number, metric):
    # Plot the p values
    rows = 1
    cols = 1
    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(6*cols, 6*rows), squeeze=False)
    ax = axes[0,0]
    im = ax.matshow(all_p_values, interpolation='none', cmap="OrRd")
    fig.colorbar(im)

    n = len(experiments)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(experiments)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(experiments)


    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45, ha="left", va="center",rotation_mode="anchor")
    fig.tight_layout()


    for (i,j), value in np.ndenumerate(all_p_values):
        text = "{:0.3f}".format(value)
        ax.text(j, i, text, ha="center", va="center")


    save_file_name = "maze_{}_{}_table.png".format(maze_number, metric)
    plt.savefig(save_file_name)


def main():
        
    # The name of the metric we should use
    metrics = ["nll", "mse"]
    # metric = "mse"

    # The mazes to compare with
    mazes_to_compare = [1, 2, 3]
    # mazes_to_compare = [1]

    # the comparisons we should make
    # comparisons = []
    # comparisons.append(("experiment0001", "diffy_particle_filter"))
    # comparisons.append(("experiment0001", "experiment0002"))
    # comparisons.append(("experiment0003", "experiment0002"))
    # comparisons.append(("experiment0003", "experiment0001"))

    # all possible pairs
    experiments = ["diffy_particle_filter", "optimal_transport_pf", "soft_resampling_pf", "experiment0001", "experiment0002", "experiment0003"]
    # experiments = ["diffy_particle_filter", "optimal_transport_pf"]
    comparisons = [[a, b] for idx, a in enumerate(experiments) for b in experiments[idx + 1:]]
    for c in comparisons:
        c.sort()

    # Create a lookup table
    index_lookup_table = dict()
    for i, e in enumerate(experiments):
         index_lookup_table[e] = i

    for metric in metrics:

        # Compare!
        for maze_number in mazes_to_compare:

            # Make a directory for that maze
            save_dir = "./maze_{}".format(maze_number)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # The title of this comparison
            comparison_title = "Deepmind Maze {} ({})".format(maze_number, metric)

            # The P values for all the different experiments
            all_p_values = np.zeros((len(experiments), len(experiments)))

            # Run through all the comparisons we want to do
            for comparison in comparisons:

                # The save file name
                save_file_name = "{}/maze_{}_{}_{}_vs_{}.png".format(save_dir, maze_number, metric, comparison[0], comparison[1])

                # make the comparison plot
                p_value = compare(comparison[0], comparison[1], maze_number, metric, comparison_title, save_file_name)

                # fill in the p value table
                i = index_lookup_table[comparison[0]]
                j = index_lookup_table[comparison[1]]
                all_p_values[i, j] = p_value
                all_p_values[j, i] = p_value


            # Plot the P values
            plot_p_values(all_p_values, experiments, maze_number, metric)

if __name__ == '__main__':
    main()