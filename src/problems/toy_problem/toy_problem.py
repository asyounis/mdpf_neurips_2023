
# Imports
import torch

# Project Imports 
from problems.problem_base import *
from datasets.toy_problem_dataset import *


class ToyProblemObservationTransformer():
    def __init__(self):
        pass

    def forward_tranform(self, observations):
        return observations

    def backward_tranform(self, observations):
        return observations

class ToyProblem(ProblemBase):
    def __init__(self, experiment, save_dir, device, dataset_types):
        super().__init__(experiment, save_dir, device, dataset_types)

        # The observation transformer that we will be using
        self.observation_transformer = ToyProblemObservationTransformer()

        # Create the datasets
        for dataset_type in dataset_types:
            dataset = ToyProblemDataset(self.dataset_params, dataset_type)
            self.add_dataset(dataset_type, dataset)


