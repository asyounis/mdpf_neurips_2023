# Standard Imports
import random
import numpy as np
import time

# Pytorch Imports
import torch



class EvaluationBase:
    def __init__(self, experiment, problem, save_dir, device, seed=0):
        
        # Save important things
        self.problem = problem
        self.save_dir = save_dir
        self.device = device

        # Get the parameters
        evaluation_params = experiment["evaluation_params"]
        
        if("random_seed" in evaluation_params):
            random_seed = evaluation_params["random_seed"]

            if(random_seed == "time"):
                random_seed = int(time.time())
            else:
                random_seed = int(random_seed)

        else:
            random_seed = seed
        
        print("Random Seed:", random_seed)


        # Set the random seed to make the evaluation reproducible
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Get the evaluation dataset
        self.evaluation_dataset = self.problem.get_evaluation_dataset()

    def run_evaluation(self):
        raise NotImplemented