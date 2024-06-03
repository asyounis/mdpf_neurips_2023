
# Imports
import torch
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable



def load_data_file(base_experiment_dir, experiment, training_loss):
    if(training_loss == "NLL"):
        evaluation_directory_postfix = ""
    elif(training_loss == "MSE"):
        evaluation_directory_postfix = "_mse"
    else:
        assert(False)

    if(experiment == "lstm_rnn"):
        results_filepath = "../../../experiments/{}/{}/saves/full_rnn_evaluation{}/table_results.pt".format(base_experiment_dir, experiment, evaluation_directory_postfix)
    else:
        results_filepath = "../../../experiments/{}/{}/saves/full_dpf_evaluation_fixed_bands{}/table_results.pt".format(base_experiment_dir, experiment, evaluation_directory_postfix)

    # Check if the file exists, if it doesnt then skip this file
    if(os.path.isfile(results_filepath) == False):

        if(training_loss == "MSE"):
            return {'nll': (-100, -100), 'mse': (-100, -100), 'rmse': (-100, -100)}

        return None

    # Load the data 
    results = torch.load(results_filepath)  


    return results


def load_all_data(base_experiment_dir, experiments, column_specifications):

    all_data_rows = dict()

    # Get the row information for each experiment
    for experiment in experiments:

        skip_this_data = False
        row_data = []

        # We need to extract the information on a per column basis 
        for col_number in column_specifications:

            # Get the specific things we need for this column
            col_spec = column_specifications[col_number]
            training_loss = col_spec["training_loss"]   
            metric = col_spec["metric"]   

            # Load the data for this column
            data = load_data_file(base_experiment_dir, experiment, training_loss)

            if(data is None):
                skip_this_data = True
                print(base_experiment_dir, experiment, training_loss)
                break

        
            # extract the specific metric data we need from this column
            metric_data = data[metric]

            # Add the data to the row
            row_data.append(metric_data)

        # save the row data if we have all the data
        if(skip_this_data == False):
            all_data_rows[experiment] = row_data

    return all_data_rows


def print_human_readable_table(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names):

    table = PrettyTable()

    ############################################################################################################
    # Generate the header
    ############################################################################################################

    field_names = ["experiment"]
    for col_number in column_specifications.keys():
        # Get the specific things we need for this column
        col_spec = column_specifications[col_number]
        training_loss = col_spec["training_loss"]   
        metric = col_spec["metric"]   

        s = "TL: {:s}, Met: {:s}".format(training_loss, metric)
        field_names.append(s)

    table.field_names = field_names



    ############################################################################################################
    # Fill in the rows
    ############################################################################################################

    for experiment in experiments:

        # If its not in then skip it
        if(experiment not in all_data_rows):
            continue

        # extract the row data
        row_data = all_data_rows[experiment]

        # Get the experiment name
        experiment_name = experiment_names[experiment]

        # create the final row that we will output
        final_row_output = [experiment_name]
        for rd in row_data:

            # Extract the mean and std so we can print them
            mean = rd[0]
            std = rd[1]

            # Create the string and add it to the table
            s = "{:0.4f} ± {:0.4f}".format(mean, std)
            final_row_output.append(s)


        table.add_row(final_row_output)



    with open("{}_human_readable.txt".format(base_experiment_dir), 'w') as f:
        f.write(table.get_string())
        f.write("\n\n\n\n\n\n\n")
        f.write(table.get_csv_string())



def print_data_latex(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names):

    # Find the best for each column
    column_bests = dict()
    for col_num in range(len(column_specifications)):

        best_val = 10000000
        best_exp_name = None

        for experiment in all_data_rows.keys():

            if(all_data_rows[experiment][col_num][0] < best_val):
                best_val = all_data_rows[experiment][col_num][0]
                best_exp_name = experiment

        column_bests[col_num] = best_exp_name


    output = "\n\n"

    # # Add the header
    output += "\\begin{table}[tb]\n"
    output += "\t\\setlength{\\tabcolsep}{2.5pt}\n"
    output += "\t\\centering\n"
    # output += "\t\\caption{caption}\n"
    output += "\t\\caption{Results of Bearings Only Tracking Task evaluated on the densely labeled evaluation (test) dataset. Methods are trained with $\\mathcal{L}_{NLL}$ and evaluated with the NLL loss on the evaluation dataset. After refinement training with $\\mathcal{L}_{MSE}$ we evaluate using the RMSE metric. Shown is the mean and standard deviation for each metric using evaluation data with best performing method in bold.}\n"
    output += "\t\\label{tab:bearings_test_results}\n"
    output += "\t\\begin{center}\n"
    output += "\t\t\\begin{small}\n"
    output += "\t\t\t\\begin{sc}\n"
    output += "\t\t\t\t\\begin{tabular}{lcc}\n"
    # output += "\t\t\t\t\t\\cmidrule[\\heavyrulewidth]{2-7}\n"
    output += "\t\t\t\t\t\\toprule\n"
    output += "\t\t\t\t\t Training Loss: &  \\multicolumn{1}{c|}{$\\mathcal{L}_{NLL}$} &  \\multicolumn{1}{|c}{$\\mathcal{L}_{MSE}$} \\\\ \n"
    output += "\t\t\t\t\t\\midrule \n"
    output += "\t\t\t\t\tMethod & NLL & RMSE \\\\ \n"
    output += "\t\t\t\t\t\\midrule \n"



    # Fill in the data
    for experiment in experiments:

        # If its not in then skip it
        if(experiment not in all_data_rows):
            continue

        # Set the experiment name
        experiment_name = experiment_names[experiment]
        output += "\t\t\t\t\t"
        output += "{0:30}".format(experiment_name) 

        # Grab the experiment row
        row_data = all_data_rows[experiment]

        # Add it
        for i in range(len(row_data)):
            rd = row_data[i]
            mean = rd[0]
            std = rd[1] 
            
            # Determine if this is the best
            is_best = False
            if(column_bests[i] == experiment):
                is_best = True

            # Add the data to the row
            output += " & "
            if(is_best):
                s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
                s = "\\textbf{" + s + "}"
            else:
                s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
            output += "{0:30}".format((s))


        output += "\\\\ \n"


    # Add the footer
    output += "\t\t\t\t\t\\bottomrule \n"
    output += "\t\t\t\t\\end{tabular} \n"
    output += "\t\t\t\\end{sc} \n"
    output += "\t\t\\end{small} \n"
    output += "\t\\end{center} \n"
    output += "\t\\vskip -0.1in \n"
    output += "\\end{table} \n"


    with open("{}_latex.txt".format(base_experiment_dir), 'w') as f:
        f.write(output)


def main():

    # The directories that we want to load data from
    BASE_EXPERIMENT_DIRS = ["toy_problem_experiments_dynamics_distribution", "toy_problem_experiments_dynamics_distribution_is"]

    # The specifications for what each column has
    # DO NOT CHANGE THIS
    column_specifications = dict()
    column_specifications[0] = {"training_loss":"NLL", "metric": "nll"}
    column_specifications[1] = {"training_loss":"MSE", "metric": "rmse"}


    # The experiments we want to load and the order we want them to be rendered in
    experiments = ["diffy_particle_filter", "optimal_transport_pf", "soft_resampling_particle_filter", "importance_sampling_pf", "experiment0001", "experiment0002_importance", "experiment0002_concrete", "experiment0003_importance", "experiment0003_concrete"]    
    experiments.extend(["diffy_particle_filter_learned_band", "optimal_transport_pf_learned_band", "soft_resampling_particle_filter_learned_band", "importance_sampling_pf_learned_band"])

    # The names we want to give to the experiments
    experiment_names = dict()
    experiment_names["lstm_rnn"] = "LSTM"
    experiment_names["diffy_particle_filter"] = "TG-PF"
    experiment_names["optimal_transport_pf"] = "OT-PF"
    experiment_names["soft_resampling_particle_filter"] = "SR-PF"
    experiment_names["importance_sampling_pf"] = "DIS-PF"

    experiment_names["experiment0001"] = "TG-MDPF"
    experiment_names["experiment0002_importance"] = "MDPF"
    experiment_names["experiment0003_importance"] = "A-MDPF"

    experiment_names["experiment0002_implicit"] = "MDPF-Implicit"
    experiment_names["experiment0003_implicit"] = "A-MDPF-Implicit"

    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"




    experiment_names["diffy_particle_filter_learned_band"] = "TG-PF-LB"
    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF-LB"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF-LB"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF-LB"



    for base_experiment_dir in BASE_EXPERIMENT_DIRS:

        # Load the data
        all_data_rows = load_all_data(base_experiment_dir, experiments, column_specifications)

        # Save the human readable table
        print_human_readable_table(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names)

        # Save the latex version
        print_data_latex(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names)


if __name__ == '__main__':
    main()