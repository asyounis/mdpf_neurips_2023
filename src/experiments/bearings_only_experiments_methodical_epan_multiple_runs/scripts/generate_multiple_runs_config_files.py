import yaml
import os
import copy

def dict_replace_item(obj, key, string_to_replace, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = dict_replace_item(v, key, string_to_replace, replace_value)
    if key in obj:
        obj[key] = obj[key].replace(string_to_replace, replace_value)
    return obj



def generate_tasks_one_large_config(tasks, number_of_trials):
    
    # All the tasks we want to run
    all_output_tasks = []

    for task in tasks:

        # Generate each task N times
        for i in range(number_of_trials):
            
            # Make a copy so we can mutate it safely
            task_copy = copy.deepcopy(task)

            # Change the key name, not needed but it makes the file look better
            name = list(task_copy.keys())[0]
            new_name = "{}_{:03d}".format(name, i)
            task_copy[new_name] = task_copy[name]
            task_copy.pop(name)

            # Mutate the save dir string
            dict_replace_item(task_copy, "save_dir", "<root_save_dir>/", "<root_save_dir>/run_{:03d}/".format(i))
            dict_replace_item(task_copy, "save_dir", "<run_number>/", "run_{:03d}/".format(i))

            # Mutate the model loading strings
            dict_replace_item(task_copy, "dpf_model", "<root_save_dir>/", "<root_save_dir>/run_{:03d}/".format(i))
            dict_replace_item(task_copy, "dpf_model", "<run_number>/", "run_{:03d}/".format(i))




            all_output_tasks.append(task_copy)

    return all_output_tasks

def process_config_file(base_config_file, number_of_trials, output_dir):

    # Load the base config file into yaml
    with open(base_config_file, 'r') as content_file:
        base_config = yaml.load(content_file, Loader=yaml.FullLoader)
    
    # Create the config file we want to output
    output_config = dict()
    output_config["experiments"] = generate_tasks_one_large_config(copy.deepcopy(base_config["experiments"]), number_of_trials)


    # The yaml file we will save
    output_config_file = "{}/{}".format(output_dir, os.path.basename(base_config_file))

    # Save!
    with open(output_config_file, 'w') as file:
        documents = yaml.dump(output_config, file)


    

def main():
    
    NUMBER_OF_TRIALS = 11
    OUTPUT_DIR = "../configs/generated/"

    # Make sure the output directory exists
    if(not os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)

    # Process all the base config directories
    base_config_files = ["../configs/our_experiments.yaml", "../configs/comparison_experiments.yaml","../configs/our_experiments_exp3_init.yaml", "../configs/lstm_experiments.yaml", "../configs/comparison_experiments_always_on_override.yaml"]
    for base_config_file in base_config_files:
        process_config_file(base_config_file, NUMBER_OF_TRIALS, OUTPUT_DIR)


if __name__ == '__main__':
    main()






