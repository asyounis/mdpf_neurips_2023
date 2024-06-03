# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# Add the packages we would like to import from to the path
import sys
sys.path.append('../packages/kernel-density-estimator-bandwdidth-prediction/packages/')

# Other useful package
import yaml
import pynvml
import socket
import time 

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F



# Project Imports
from trainers.training_runner import *
from evaluations.evaluations_runner import *
from data_generators.data_generator_runner import *
from models.model_creator import *



# Change the way pytorch prints things
torch.set_printoptions(precision=8, linewidth=600, sci_mode=False)
# torch.set_printoptions(precision=8, linewidth=600, sci_mode=False, threshold=10_000)

# Enable cuDNN optimizations
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False

# print("torch.autograd.set_detect_anomaly(True)")
# torch.autograd.set_detect_anomaly(True)

def get_gpu_info(device):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    num_compute_processes_running = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
        
    pynvml.nvmlShutdown()

    return mem_info.free, num_compute_processes_running


def get_gpu_with_most_free_ram():

    # Get the number of devices
    device_count = torch.cuda.device_count()

    # Get the labels for each of the devices
    if("CUDA_VISIBLE_DEVICES" in os.environ):
        all_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        all_devices = [int(s) for s in all_devices]
    else:
       all_devices = [i for i in range(device_count)] 


    free_ram_value = 0
    num_processes_value = 100000

    gpu_select_index = -1

    for i in range(device_count):

        device_id = all_devices[i]

        free_ram, num_compute_processes_running = get_gpu_info(device_id)
        print("GPU {}  has {} MiB free".format(i, (free_ram / (1024*1024))))

        if(abs(free_ram_value - free_ram) < (500 * 1024 * 1024)):
            if(num_compute_processes_running < num_processes_value):
                gpu_select_index = i;
                free_ram_value = free_ram
                num_processes_value = num_compute_processes_running
        else:
            if(free_ram_value < free_ram):
                gpu_select_index = i;
                free_ram_value = free_ram
                num_processes_value = num_compute_processes_running

    return gpu_select_index

def recursive_update(target, new_stuff):


    for key in new_stuff.keys():

        if(key not in target):
            target[key] = new_stuff[key]
        else:
            if(isinstance(new_stuff[key], dict)):
                assert(isinstance(target[key], dict))
                target[key] = recursive_update(target[key], new_stuff[key])
            else:
                target[key] = new_stuff[key]   

    return target


def load_common_parameters_file(common_parameters_file, experiment):

    # Read and parse the config file
    with open(common_parameters_file) as file:

        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        return recursive_update(doc, experiment)


def run_experiment(experiment_name, experiment, model_architecture_params, save_dir, device, save_dir_root, experiment_name_replacement):
    # Delete the save file since it gets very long and so we should just delete at the beginning of every run
    if os.path.exists(save_dir + "/output.txt"):
        os.remove(save_dir + "/output.txt")

    # Create an output file that will hold some useful info about this experiment
    output_file = open(save_dir + "/output.txt", "a")
    output_file.write(experiment_name + "\n")
    output_file.write("Running on device: {}\n".format(device))
    output_file.write("\n============================================================================\n")
    output_file.close()

    print("")
    print("")
    print("==============================================================")
    print("Running \"{}\"".format(experiment_name))
    print("==============================================================")

    print("Running on device: {}".format(device))

    # Create the model that we will be using for this experiment
    model_params = experiment["model_params"]
    model = create_model(model_params, model_architecture_params)

    did_load_pretrained_model = False
    # Load the model if we have anything to load
    if(("pre_trained_models_local_override" in experiment) or ("pre_trained_models" in experiment)):
        
        if("pre_trained_models_local_override" in experiment):
            pre_trained_models = experiment["pre_trained_models_local_override"]
        elif("pre_trained_models" in experiment):
            pre_trained_models = experiment["pre_trained_models"]
        else:
            assert(False)


        # Fill in with the right path if we are using the root save dir
        for model_name in list(pre_trained_models.keys()):
            model_path = pre_trained_models[model_name]

            # If we have the root save dir tag then string replace it with the correct thing
            if("<root_save_dir>" in model_path):
                model_path = model_path.replace("<root_save_dir>", save_dir_root)
                pre_trained_models[model_name] = model_path

            if("<experiment_name_replacement>" in model_path):
                model_path = model_path.replace("<experiment_name_replacement>", experiment_name_replacement)
                pre_trained_models[model_name] = model_path


        model.load_pretrained(pre_trained_models, device)
        did_load_pretrained_model = True

    else:
        print("\"pre_trained_models\" not specified. Not loading any pre-trained model weights!")


    # Sometimes we want to scale the bandwidths after loading them from save files (aka when we are converting from gauss to epan).
    if("scale_bandwidths_on_init_params" in experiment):
        if(did_load_pretrained_model == False):
            print("Cant scale bandwidths on init if you are not loading models from save...")
            print("Just make the bandwidths bigger in the model file by default...")
            assert(False)

        # Get the scale params
        scale_bandwidths_on_init_params = experiment["scale_bandwidths_on_init_params"]

        # Scale
        model.scale_bandwidths_on_init(scale_bandwidths_on_init_params)


    # Move the model to the correct device
    model = model.to(device)

    # Extract the experiment type
    experiement_type = experiment["experiement_type"]
    print("Experiment Type: {}".format(experiement_type))

    # Switch between training and evaluation
    if(experiement_type == "training"):
        setup_training(experiment, model, save_dir, device)
    elif(experiement_type == "evaluation"):
        setup_evaluation(experiment, model, save_dir, device)
    elif(experiement_type == "data_generator"):
        setup_data_generator(experiment, model, save_dir, device)
    else:
        print("Unknown experiement_type... Exiting")
        exit()                


def load_experiments_from_experiments_import_files(experiments_import):

    experiments = []

    for filepath in experiments_import:

        # Read and parse the file
        with open(filepath) as file:

            # Load the whole file into a dictionary
            doc = yaml.load(file, Loader=yaml.FullLoader)

            # Make sure there is nothing in this file other than the experiments root 
            assert(len(doc.keys()) <= 1)
            assert(list(doc.keys())[0] == "experiments")

            # append the experiments
            experiments.extend(doc["experiments"])

    return experiments

def make_save_dir(save_dir_local, save_dir_root):
    if(len(save_dir_root) == 0):
        return save_dir_local

    if("<root_save_dir>" in save_dir_local):
        return save_dir_local.replace("<root_save_dir>", save_dir_root)
    
    return save_dir_local

def main():

    # # Set the random seed to make the evaluation reproducible
    # seed = 15
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)


    # Get the config file from the command line arguments
    config_file = sys.argv[1]

    # Find the GPU with the most amount of memory available
    device_idx = get_gpu_with_most_free_ram()
    device_most_free_memory_at_startup = "cuda:{}".format(device_idx)

    # Read and parse the config file
    with open(config_file) as file:
        
        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # The arch params for all the models
        model_architecture_params = dict()

        # Load common model files if present:
        if("model_files" in doc):
            for f in doc["model_files"]:
                with open(f) as m_file:
                    models_doc = yaml.load(m_file, Loader=yaml.FullLoader)
                    model_architecture_params.update(models_doc["models"])               

        # Load the model parameters
        model_architecture_params.update(doc["models"])              

        # Load the save root if one is present
        if("save_dir_root" in doc):
            save_dir_root = doc["save_dir_root"]
        else:
            save_dir_root = ""

        # Load the experiment_name_replacement if one is present
        if("experiment_name_replacement" in doc):
            experiment_name_replacement = doc["experiment_name_replacement"]
        else:
            experiment_name_replacement = ""

        # Load any experiment imports if we have them
        if("experiments_import" in doc):
            experiments = load_experiments_from_experiments_import_files(doc["experiments_import"])
        else:
            experiments = []

        # Load the local experiments and append them to the current set of experiment we wish to run
        if("experiments" in doc):
            experiments.extend(doc["experiments"])

        if("experiments_local_override" in doc):
            experiments_local_override = doc["experiments_local_override"]

            # Repackage into dict
            tmp = dict()
            for e in experiments_local_override:
                name = list(e.keys())[0]
                assert(name not in tmp)
                tmp[name] = e[name] 

            experiments_local_override = tmp
        else:
            experiments_local_override = None


        # Read each experiment 1 at a time
        for experiment_params in experiments:

            # Extract the Experiment Name
            experiment_name = list(experiment_params.keys())[0]
            experiment = experiment_params[experiment_name]

            # Get the common parameters file and load it if we have one
            if("common_parameters_file" in experiment):
                common_parameters_file = experiment["common_parameters_file"]

                # Load the common parameters file and update the experiment with the new settings
                experiment = load_common_parameters_file(common_parameters_file, experiment)

            # Make any updates using local overrides
            if((experiments_local_override is not None) and (experiment_name in experiments_local_override)):
                local_override = experiments_local_override[experiment_name]
                experiment = recursive_update(experiment, local_override)


            # Extract if we should run this experiment or not.  If we dont have it then by default we want to run
            # otherwise check the parameter and see what its value is
            if("do_run" in experiment):
                do_run = experiment["do_run"]
                if(do_run == False):
                    continue

            # Extract and create  the save directory.  This is where all the output artifacts will be places
            save_dir = make_save_dir(experiment["save_dir"], save_dir_root)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Extract the device to use
            device = experiment["device"]
            if(device == "cuda_auto"):
                device = device_most_free_memory_at_startup
                
            elif(device == "cuda_auto_balance"):
                # Find the GPU with the most amount of memory available
                device_idx = get_gpu_with_most_free_ram()
                device_most_free_memory_at_startup = "cuda:{}".format(device_idx)
                device = device_most_free_memory_at_startup

            if("cuda" in device):
                with torch.cuda.device(device):
                    run_experiment(experiment_name, experiment, model_architecture_params, save_dir, device, save_dir_root, experiment_name_replacement)
            else:
                run_experiment(experiment_name, experiment, model_architecture_params, save_dir, device, save_dir_root, experiment_name_replacement)


if __name__ == '__main__':
    main()


# # Do this to hopefully fix the hanging issue with data loaders
# # https://github.com/pytorch/pytorch/issues/1355
# # 
# # Also maybe need to do this:
# #   sudo su
# #   echo "16384" > /proc/sys/kernel/shmmni
# import multiprocessing as mp
# if __name__ == "__main__":
#     mp.set_start_method('spawn')
#     main()
