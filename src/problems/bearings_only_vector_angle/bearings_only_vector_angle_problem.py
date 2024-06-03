
# Imports
import torch

# Project Imports 
from problems.problem_base import *
from datasets.bearings_only_vector_angle_dataset import *


class BearingsOnlyVectorAngleObservationTransformer():
    def __init__(self):
        pass

    def forward_tranform(self, observations):
        
        sin = torch.sin(observations)
        cos = torch.cos(observations)

        return torch.cat([sin, cos], dim=-1)

    def backward_tranform(self, observations):

        last_dim_half_size = int(observations.shape[-1] / 2)

        sin = observations[..., :last_dim_half_size]
        cos = observations[..., last_dim_half_size:]

        theta = torch.atan2(sin, cos)

        return theta


class BearingsOnlyVectorAngleProblem(ProblemBase):
    def __init__(self, experiment, save_dir, device, dataset_types):
        super().__init__(experiment, save_dir, device, dataset_types)

        # The observation transformer that we will be using
        self.observation_transformer = BearingsOnlyVectorAngleObservationTransformer()

        # Create the datasets
        for dataset_type in dataset_types:
            dataset = BearingsOnlyVectorAngleDataset(self.dataset_params, dataset_type)
            self.add_dataset(dataset_type, dataset)


