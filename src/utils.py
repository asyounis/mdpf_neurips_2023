# Imports
import torch
import numpy as np

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def get_parameter_safely( param_name, data_dict, data_dict_name):
    if(param_name not in data_dict):
        print("Could not find \"{}\" in \"{}\"".format(param_name, data_dict_name))
        assert(False)

    return data_dict[param_name]
