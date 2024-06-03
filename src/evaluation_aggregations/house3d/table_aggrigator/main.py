
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

    results_filepath = "../../../experiments/{}/{}/saves/full_dpf_evaluation_fixed_bands{}/table_results.pt".format(base_experiment_dir, experiment, evaluation_directory_postfix)

    # Check if the file exists, if it doesnt then skip this file
    if(os.path.isfile(results_filepath) == False):

        if(training_loss == "MSE"):
            return {'nll': (-100, -100), 'mse': (-100, -100), 'rmse': (-100, -100)}

        print(results_filepath)
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
            iqr = rd[2]

            # Create the string and add it to the table
            s = "{:0.4f} ± {:0.4f}, {:0.4f}".format(mean, std, iqr)
            final_row_output.append(s)


        table.add_row(final_row_output)



    with open("./human_readable/{}_human_readable.txt".format(base_experiment_dir), 'w') as f:
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
    output += "\t\\caption{}\n"
    output += "\t\\label{tab:house_3d_results_table}\n"
    output += "\t\\begin{center}\n"
    output += "\t\t\\begin{small}\n"
    output += "\t\t\t\\begin{sc}\n"
    output += "\t\t\t\t\\begin{tabular}{lcc}\n"
    # output += "\t\t\t\t\t\\cmidrule[\\heavyrulewidth]{2-7}\n"
    output += "\t\t\t\t\t\\toprule\n"
    # output += "\t\t\t\t\t Training Loss: &  \\multicolumn{1}{c|}{$\\mathcal{L}_{NLL}$} &  \\multicolumn{1}{|c}{$\\mathcal{L}_{MSE}$} \\\\ \n"
    # output += "\t\t\t\t\t\\midrule \n"
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
            iqr = rd[2]

            # Determine if this is the best
            is_best = False
            if(column_bests[i] == experiment):
                is_best = True

            # Add the data to the row
            output += " & "
            if(is_best):
                s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, iqr)
                s = "\\textbf{" + s + "}"
            else:
                s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, iqr)
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


    with open("./latex/{}_latex.txt".format(base_experiment_dir), 'w') as f:
        f.write(output)



# def print_data_latex(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names):

#     # Find the best for each column
#     column_bests = dict()
#     for col_num in range(len(column_specifications)):

#         best_val = 10000000
#         best_exp_name = None

#         for experiment in all_data_rows.keys():

#             if(all_data_rows[experiment][col_num][0] < best_val):
#                 best_val = all_data_rows[experiment][col_num][0]
#                 best_exp_name = experiment

#         column_bests[col_num] = best_exp_name


#     # Filter out the experiments
#     filtered_experiments = []
#     for experiment in experiments:
#         if(experiment in all_data_rows):
#             filtered_experiments.append(experiment)


#     # 1 Column per experiment and 1 col for the row labels
#     number_of_cols = len(filtered_experiments) + 1

#     # Compute the tabular column setting string
#     tabular_col_string = "l" + "".join(["c" for _ in range(len(filtered_experiments))])



#     output = "\n\n"

#     # # Add the header
#     output += "\\begin{table}[tb]\n"
#     output += "\t\\setlength{\\tabcolsep}{2.5pt}\n"
#     output += "\t\\centering\n"
#     # output += "\t\\caption{caption}\n"
#     output += "\t\\caption{}\n"
#     output += "\t\\label{tab:house_3d_results_table}\n"
#     output += "\t\\begin{center}\n"
#     output += "\t\t\\begin{small}\n"
#     output += "\t\t\t\\begin{sc}\n"
#     output += "\t\t\t\t\\begin{tabular}{" + tabular_col_string + "}\n"
#     output += "\t\t\t\t\t\\toprule\n"
#     # output += "\t\t\t\t\tMethod & NLL & RMSE \\\\ \n"



#     row_output = "\t\t\t\t\t Method"
#     for experiment in filtered_experiments:
#         row_output += " & " + experiment_names[experiment]
#     row_output += "\\\\ \n"
#     output += row_output
#     output += "\t\t\t\t\t\\midrule \n"



#     row_output = "\t\t\t\t\t NLL"
#     for experiment in filtered_experiments:
#         row_output += " & "

#         mean = all_data_rows[experiment][0][0]
#         std = all_data_rows[experiment][0][1]

#         # Determine if this is the best
#         is_best = False
#         if(column_bests[0] == experiment):
#             is_best = True

#         # Add the data to the row
#         if(is_best):
#             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#             s = "\\textbf{" + s + "}"
#         else:
#             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#         row_output += s


#     row_output += "\\\\ \n"
#     output += row_output
#     output += "\t\t\t\t\t\\midrule \n"




#     row_output = "\t\t\t\t\t RMSE"
#     for experiment in filtered_experiments:
#         row_output += " & "

#         mean = all_data_rows[experiment][1][0]
#         std = all_data_rows[experiment][1][1]

#         # Determine if this is the best
#         is_best = False
#         if(column_bests[1] == experiment):
#             is_best = True

#         # Add the data to the row
#         if(is_best):
#             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#             s = "\\textbf{" + s + "}"
#         else:
#             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#         row_output += s


#     row_output += "\\\\ \n"
#     output += row_output
#     output += "\t\t\t\t\t\\midrule \n"


#     # print(all_data_rows[filtered_experiments[0]])
#     # exit()


#     # # Fill in the data
#     # for experiment in filtered_experiments:

#     #     # Set the experiment name
#     #     experiment_name = experiment_names[experiment]
#     #     output += "\t\t\t\t\t"
#     #     output += "{0:30}".format(experiment_name) 

#     #     # Grab the experiment row
#     #     row_data = all_data_rows[experiment]

#     #     # Add it
#     #     for i in range(len(row_data)):
#     #         rd = row_data[i]
#     #         mean = rd[0]
#     #         std = rd[1] 
            
#     #         # Determine if this is the best
#     #         is_best = False
#     #         if(column_bests[i] == experiment):
#     #             is_best = True

#     #         # Add the data to the row
#     #         output += " & "
#     #         if(is_best):
#     #             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#     #             s = "\\textbf{" + s + "}"
#     #         else:
#     #             s = "{:0.2f} $\\pm$ {:0.2f}".format(mean, std)
#     #         output += "{0:30}".format((s))


#         # output += "\\\\ \n"


#     # Add the footer
#     output += "\t\t\t\t\t\\bottomrule \n"
#     output += "\t\t\t\t\\end{tabular} \n"
#     output += "\t\t\t\\end{sc} \n"
#     output += "\t\t\\end{small} \n"
#     output += "\t\\end{center} \n"
#     output += "\t\\vskip -0.1in \n"
#     output += "\\end{table} \n"


#     with open("./latex/{}_latex.txt".format(base_experiment_dir), 'w') as f:
#         f.write(output)





def get_name_mapping():

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
    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF"
    experiment_names["discrete_concrete"] = "C-PF"

    return experiment_names


def main():

    # The directories that we want to load data from
    BASE_EXPERIMENT_DIRS = []
    BASE_EXPERIMENT_DIRS.append("house3d_experiments")


    # The specifications for what each column has
    # DO NOT CHANGE THIS
    column_specifications = dict()
    column_specifications[0] = {"training_loss":"NLL", "metric": "nll"}
    column_specifications[1] = {"training_loss":"MSE", "metric": "rmse"}



    experiments = []
    experiments.append("lstm_rnn")
    experiments.append("diffy_particle_filter_learned_band")
    experiments.append("optimal_transport_pf_learned_band")
    experiments.append("soft_resampling_particle_filter_learned_band")
    experiments.append("importance_sampling_pf_learned_band")
    experiments.append("discrete_concrete")
    experiments.append("experiment0002_concrete")
    experiments.append("experiment0003_concrete")
    experiments.append("experiment0001")
    experiments.append("experiment0002_importance")
    # experiments.append("experiment0003_importance")
    experiments.append("experiment0003_importance_init")




    # The names we want to give to the experiments
    experiment_names = get_name_mapping()


    if(os.path.exists("./latex/") == False):
        os.makedirs("./latex/")

    if(os.path.exists("./human_readable/") == False):
        os.makedirs("./human_readable/")


    for base_experiment_dir in BASE_EXPERIMENT_DIRS:

        # Load the data
        all_data_rows = load_all_data(base_experiment_dir, experiments, column_specifications)

        # Save the human readable table
        print_human_readable_table(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names)

        # Save the latex version
        print_data_latex(base_experiment_dir, all_data_rows, experiments, column_specifications, experiment_names)


if __name__ == '__main__':
    main()