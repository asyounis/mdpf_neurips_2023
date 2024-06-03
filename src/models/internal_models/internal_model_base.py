

# Pytorch Imports
import torch
import torch.nn as nn



class InternalModelBase(nn.Module):
    def __init__(self, model_parameters):
        super(InternalModelBase, self).__init__()


    def is_learned_model(self): 
        ''' Returns True if this is a learned model and not just a model that
            Does some math but is not learned.  This will matter since we dont add
            non-learned models to the trainer for saving and we dont create optimizers for non learned models
        '''
        raise NotImplemented

class NonLearnedInternalModelBase(InternalModelBase):
    def __init__(self, model_parameters):
        super(NonLearnedInternalModelBase, self).__init__(model_parameters)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def is_learned_model(self): 
        ''' Returns True if this is a learned model and not just a model that
            Does some math but is not learned.  This will matter since we dont add
            non-learned models to the trainer for saving and we dont create optimizers for non learned models
        '''
        return False


    def load_state_dict(self, state_dict, strict=True):
        ''' Do not load anything.  This is a hack since we could maybe try to load a saved model file into a non learned model for reasons...
        '''
        pass


class LearnedInternalModelBase(InternalModelBase):
    def __init__(self, model_parameters):
        super(LearnedInternalModelBase, self).__init__(model_parameters)

        # Extract if we should use spectral norms or not
        if("use_spectral_norm" in model_parameters):
            self.use_spectral_norm = model_parameters["use_spectral_norm"]
        else:
            self.use_spectral_norm = False

        # Extract if we should use weight norm parameterization
        if("use_weight_norm" in model_parameters):
            self.use_weight_norm = model_parameters["use_weight_norm"]
        else:
            self.use_weight_norm = False


    def apply_parameter_norm(self, module):

        module = self.apply_spectral_norm(module)
        module = self.apply_weight_norm(module)
        
        return module


    def apply_spectral_norm(self, module):

        # Spectral norm is on so apply the norm
        if(self.use_spectral_norm):
            return nn.utils.parametrizations.spectral_norm(module)

        # We are not using spectral norm so return the same module
        return module


    def apply_weight_norm(self, module):

        # Weight norm is on so apply the norm
        if(self.use_weight_norm):
            return nn.utils.weight_norm(module)

        # We are not using weight norm so return the same module
        return module

    def freeze_batchnorms(self):

        for module in self.modules():
            if(isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):

                # If we have the weights and biases then we need to freeze them
                if(hasattr(module, 'weight')):
                    module.weight.requires_grad_(False)
                if(hasattr(module, 'bias')):
                    module.bias.requires_grad_(False)

                # Stop tracking the stats 
                module.track_running_stats = False 

                # Set the model to eval mode
                module.eval()


    def is_learned_model(self): 
        ''' Returns True if this is a learned model and not just a model that
            Does some math but is not learned.  This will matter since we dont add
            non-learned models to the trainer for saving and we dont create optimizers for non learned models
        '''
        return True




class ModuleUseOnlyWrapper:
    '''
        This wrapper wraps a torch Module so that it can be passed into another torch Module and used without
        having its parameters added to the main Module 
    '''
    def __init__(self, m):
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.m(*args, **kwargs)