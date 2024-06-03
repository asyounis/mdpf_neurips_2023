
#Standard Imports 
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# Project imports
from trainers.trainer_utils import *
from utils import *

class DataGeneratorBase():
    def __init__(self, params, main_model, problem, save_dir, device):
            
        # Keep track of some of the important things
        self.main_model = main_model
        self.save_dir = save_dir
        self.device = device
        self.problem = problem

        # Parse the command arguments
        self.data_generator_params = params["data_generator_params"]


    def generate_data(self):
        raise NotImplemented