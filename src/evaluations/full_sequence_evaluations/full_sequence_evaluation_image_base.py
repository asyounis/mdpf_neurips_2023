# Standard Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

# Pytorch Imports
import torch

# Project Imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_base import *

class FullSequenceEvaluationImageBase(FullSequenceEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def render_sequence(self, render_index, data, output_dicts):

        # For now we want just 1 col.  Maybe we can make this prettier later
        rows, cols = self.get_rows_and_cols()
        
        # Make the figure
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 12), squeeze=False)
        # axes = axes.reshape(-1,)

        # Render the data
        self.render_data(data, output_dicts, axes)

        fig.tight_layout(pad=0.25)

        # Save the figure
        plt.savefig("{}/renderings_{:04d}.png".format(self.save_dir, render_index))
