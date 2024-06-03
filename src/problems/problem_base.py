
# Python Imports
import yaml

# Project Imports
from datasets.bearings_only_dataset import *

class ProblemBase:
    def __init__(self, experiment, save_dir, device, dataset_types):

        # Save things that are important
        self.save_dir = save_dir
        self.device = device

        # extract the dataset parameters
        self.dataset_params = self.get_dataset_params(experiment)

        # The observation transformer that we will be using
        # Must be set by child class
        self.observation_transformer = None

        # The datasets that we created
        self.datasets = dict()

    def add_dataset(self, dataset_name, dataset):

        # Make sure we dont override an existing dataset
        if(dataset_name in self.datasets):
            print("Cant override dataset name: {}".format(dataset_name))
            assert(False)

        # Add the dataset
        self.datasets[dataset_name] = dataset

    def get_dataset(self, dataset_name):
        assert(dataset_name in self.datasets)
        return self.datasets[dataset_name]

    def get_training_dataset(self):
        return self.get_dataset("training")

    def get_validation_dataset(self):
        return self.get_dataset("validation")

    def get_evaluation_dataset(self):

        if("evaluation" not in self.datasets):
            return None
        return self.get_dataset("evaluation")

    def get_dataset_params(self, experiment):
                
        # Make sure that we dont have parameters and a file to load.  Its 1 or the other
        assert(not (("dataset_params" in experiment) and ("dataset_params_file" in experiment)))
        assert(("dataset_params" in experiment) or ("dataset_params_file" in experiment))

        # If its present then extract and return
        if("dataset_params" in experiment):
            return experiment["dataset_params"]
        else:
            # There are no dataset params so we must have a file to load, so lets load it
            dataset_params_file = experiment["dataset_params_file"]

            # Open the file and get the params
            with open(dataset_params_file) as m_file:
                doc = yaml.load(m_file, Loader=yaml.FullLoader)
                return doc["dataset_params"]              



