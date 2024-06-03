
import torch

class LossFunctionBase:

    def __init__(self, model):
        
        # Save the model so we can access things from it
        self.model = model

    def compute_loss(self, output_dict, states):
        raise NotImplemented

    def do_final_aggrigation(self, data):
        # return torch.mean(data)
        return data