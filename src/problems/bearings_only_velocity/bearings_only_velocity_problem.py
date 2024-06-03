
# Imports
import torch

# Project Imports 
from problems.problem_base import *
from datasets.bearings_only_velocity_dataset import *


class BearingsOnlyVelocityObservationTransformer():
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


class BearingsOnlyVelocityProblem(ProblemBase):
    def __init__(self, experiment, save_dir, device, dataset_types):
        super().__init__(experiment, save_dir, device, dataset_types)

        # The observation transformer that we will be using
        self.observation_transformer = BearingsOnlyVelocityObservationTransformer()

        # Create the datasets
        for dataset_type in dataset_types:
            dataset = BearingsOnlyDatasetVelocity(self.dataset_params, dataset_type)
            self.add_dataset(dataset_type, dataset)


