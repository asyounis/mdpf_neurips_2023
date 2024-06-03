# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision import transforms


# Project Imports
from models.internal_models.internal_model_base import *

class VectorObservationEncoder(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(VectorObservationEncoder, self).__init__(model_parameters)

        # Create the encoder
        self.create_encoder(model_parameters)

    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_layers" in model_parameters)
        assert("encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)
        assert("input_dimension" in model_parameters)

        # Extract the parameters needed or the timestep encoder
        self.encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_layers = model_parameters["encoder_number_of_layers"]
        encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]
        input_dimension = model_parameters["input_dimension"]

        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        # Need at least 2 layers, the input and output layers
        assert(encoder_number_of_layers >= 1)

        # Create the timestamp encoder layers
        layers = []
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=input_dimension,out_features=self.encoder_latent_space)))
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(self.encoder_latent_space))
        
        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_layers-1):
            
            layers.append(non_linear_object())
            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))

            layers.append(self.apply_parameter_norm(nn.Linear(in_features=self.encoder_latent_space,out_features=self.encoder_latent_space)))
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))


        # Generate the model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ImageObservationEncoder(LearnedInternalModelBase):

    def __init__(self, model_parameters):
        super(ImageObservationEncoder, self).__init__(model_parameters)

        # Create the encoder
        self.create_encoder(model_parameters)

    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_fc_layers" in model_parameters)
        assert("encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)
        assert("input_image_size" in model_parameters)
        assert("input_image_dim" in model_parameters)
        assert("convolution_latent_spaces" in model_parameters)
        assert("convolution_kernel_sizes" in model_parameters)


        # Extract the parameters needed
        self.encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_fc_layers = model_parameters["encoder_number_of_fc_layers"]
        encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]
        self.input_image_size = model_parameters["input_image_size"]
        input_image_dim = model_parameters["input_image_dim"]
        convolution_latent_spaces = model_parameters["convolution_latent_spaces"]
        convolution_kernel_sizes = model_parameters["convolution_kernel_sizes"]

        assert(len(convolution_latent_spaces) == len(convolution_kernel_sizes))


        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        # Need at least 1 layer
        assert(encoder_number_of_fc_layers > 0)

        layers = []

        last_channels = input_image_dim
        for i in range(len(convolution_latent_spaces)):
            conv_latent_space = convolution_latent_spaces[i]
            kernel_size = convolution_kernel_sizes[i]

            padding = int(kernel_size/2)

            layers.append(self.apply_parameter_norm(nn.Conv2d(last_channels, conv_latent_space, kernel_size, padding=padding)))

            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            layers.append(non_linear_object())

            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            last_channels = conv_latent_space

        layers.append(nn.Flatten())

        fc_input_layers = self.input_image_size*self.input_image_size*last_channels


        # Create the timestamp encoder layers
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=fc_input_layers,out_features=self.encoder_latent_space)))
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(self.encoder_latent_space))
        
        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_fc_layers-1):
            
            layers.append(non_linear_object())
            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))

            layers.append(self.apply_parameter_norm(nn.Linear(in_features=self.encoder_latent_space,out_features=self.encoder_latent_space)))
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))


        # Generate the model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ResnetObservationEncoder(LearnedInternalModelBase):

    def __init__(self, model_parameters):
        super(ResnetObservationEncoder, self).__init__(model_parameters)

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        encoder_latent_space = model_parameters["encoder_latent_space"]

        # Get the resnet model
        # Note we are getting the smallest one here!
        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # Change the output layer
        resnet_linear_input = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(resnet_linear_input, encoder_latent_space)

        # The image converter to be able to use resnet
        self.image_preprocessor = transforms.Compose([
            transforms.Resize(224), # Resnet expects this size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("encoder_number_of_fc_layers" in model_parameters)
        assert("encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)
        assert("input_image_size" in model_parameters)
        assert("input_image_dim" in model_parameters)
        assert("convolution_latent_spaces" in model_parameters)
        assert("convolution_kernel_sizes" in model_parameters)


        # Extract the parameters needed
        encoder_number_of_fc_layers = model_parameters["encoder_number_of_fc_layers"]
        encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]
        input_image_size = model_parameters["input_image_size"]
        input_image_dim = model_parameters["input_image_dim"]
        convolution_latent_spaces = model_parameters["convolution_latent_spaces"]
        convolution_kernel_sizes = model_parameters["convolution_kernel_sizes"]

        assert(len(convolution_latent_spaces) == len(convolution_kernel_sizes))


        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        # Need at least 1 layer
        assert(encoder_number_of_fc_layers > 0)

        layers = []

        last_channels = input_image_dim
        for i in range(len(convolution_latent_spaces)):
            conv_latent_space = convolution_latent_spaces[i]
            kernel_size = convolution_kernel_sizes[i]

            padding = int(kernel_size/2)

            layers.append(self.apply_parameter_norm(nn.Conv2d(last_channels, conv_latent_space, kernel_size, padding=padding)))

            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            layers.append(non_linear_object())

            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            last_channels = conv_latent_space

        layers.append(nn.Flatten())

        fc_input_layers = input_image_size*input_image_size*last_channels


        # Create the timestamp encoder layers
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=fc_input_layers,out_features=self.encoder_latent_space)))
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(self.encoder_latent_space))
        
        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_fc_layers-1):
            
            layers.append(non_linear_object())
            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))

            layers.append(self.apply_parameter_norm(nn.Linear(in_features=self.encoder_latent_space,out_features=self.encoder_latent_space)))
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))


        # Generate the model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):

        x = self.image_preprocessor(x)

        return self.resnet_model(x)




class ImageObservationWithPoolingEncoder(LearnedInternalModelBase):

    def __init__(self, model_parameters):
        super(ImageObservationWithPoolingEncoder, self).__init__(model_parameters)

        # Create the encoder
        self.create_encoder(model_parameters)

    def create_encoder(self, model_parameters):

        # Make sure we have the correct parameters are passed in
        assert("encoder_latent_space" in model_parameters)
        assert("encoder_number_of_fc_layers" in model_parameters)
        assert("encoder_use_batch_norm" in model_parameters)
        assert("non_linear_type" in model_parameters)
        assert("input_image_size" in model_parameters)
        assert("input_image_dim" in model_parameters)
        assert("convolution_latent_spaces" in model_parameters)
        assert("convolution_kernel_sizes" in model_parameters)


        # Extract the parameters needed
        self.encoder_latent_space = model_parameters["encoder_latent_space"]
        encoder_number_of_fc_layers = model_parameters["encoder_number_of_fc_layers"]
        encoder_use_batch_norm = model_parameters["encoder_use_batch_norm"]
        non_linear_type = model_parameters["non_linear_type"]
        self.input_image_size = model_parameters["input_image_size"]
        input_image_dim = model_parameters["input_image_dim"]
        convolution_latent_spaces = model_parameters["convolution_latent_spaces"]
        convolution_kernel_sizes = model_parameters["convolution_kernel_sizes"]
        pool_after_convolution = model_parameters["pool_after_convolution"]

        assert(len(convolution_latent_spaces) == len(convolution_kernel_sizes))


        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        # Need at least 1 layer
        assert(encoder_number_of_fc_layers > 0)

        layers = []

        current_image_size = self.input_image_size;

        last_channels = input_image_dim
        for i in range(len(convolution_latent_spaces)):
            conv_latent_space = convolution_latent_spaces[i]
            kernel_size = convolution_kernel_sizes[i]

            padding = int(kernel_size/2)

            layers.append(self.apply_parameter_norm(nn.Conv2d(last_channels, conv_latent_space, kernel_size, padding=padding)))

            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            layers.append(non_linear_object())

            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm2d(conv_latent_space))

            if(pool_after_convolution[i]):
                kernel_size = 3
                stride = 2
                layers.append(nn.MaxPool2d(kernel_size=(kernel_size, kernel_size), stride=(stride, stride)))
                
                # Computed using equations from:
                #   https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
                # Note: Padding = 0, Dilation = 1
                current_image_size -= (kernel_size-1)
                current_image_size -= 1
                current_image_size //= stride
                current_image_size += 1

            # Old dumb way
            # if(pool_after_convolution[i]):
            #     layers.append(nn.MaxPool2d(kernel_size=2))
            #     current_image_size //= 2

            last_channels = conv_latent_space

        layers.append(nn.Flatten())

        fc_input_layers = current_image_size*current_image_size*last_channels

        # Create the timestamp encoder layers
        layers.append(self.apply_parameter_norm(nn.Linear(in_features=fc_input_layers,out_features=self.encoder_latent_space)))
        if(encoder_use_batch_norm == "pre_activation"):
            layers.append(nn.BatchNorm1d(self.encoder_latent_space))
        
        # the middle layers are all the same fully connected layers
        for i in range(encoder_number_of_fc_layers-1):
            
            layers.append(non_linear_object())
            if(encoder_use_batch_norm == "post_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))

            layers.append(self.apply_parameter_norm(nn.Linear(in_features=self.encoder_latent_space,out_features=self.encoder_latent_space)))
            if(encoder_use_batch_norm == "pre_activation"):
                layers.append(nn.BatchNorm1d(self.encoder_latent_space))

        # Generate the model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)






class House3DObservationEncoder(LearnedInternalModelBase):

    def __init__(self, model_parameters):
        super(House3DObservationEncoder, self).__init__(model_parameters)

        # Make sure we have the correct parameters are passed in
        assert("non_linear_type" in model_parameters)

        # Extract the parameters needed
        self.non_linear_type = model_parameters["non_linear_type"]


        # Create the initial conv layers
        self.conv_layers_1 = nn.ModuleList()
        self.conv_layers_1.append(self.make_conv_layer(3, 128, (3, 3), dilation=1))
        self.conv_layers_1.append(self.make_conv_layer(3, 128, (5, 5), dilation=1))
        self.conv_layers_1.append(self.make_conv_layer(3, 64, (5, 5), dilation=2))
        self.conv_layers_1.append(self.make_conv_layer(3, 64, (5, 5), dilation=4))

        # The second convolution layers
        self.conv_layers_2 = nn.ModuleList()
        self.conv_layers_2.append(self.make_conv_layer(384, 16, (3, 3), dilation=1, use_activation=False))

        # The max pooling layers
        self.max_pooling_layer_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.max_pooling_layer_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        # The activations
        self.activation_1 = self.get_activation_object()
        self.activation_2 = self.get_activation_object()


    def make_conv_layer(self, in_channels, out_channels, kernel_size, dilation, padding="same", use_activation=True):

        layers = []

        # Make the convolution layer
        layers.append(self.apply_parameter_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)))

        # Add the non_linearity
        if(use_activation):
            layers.append(self.get_activation_object())

        # Generate the layers
        return nn.Sequential(*layers)

    def get_activation_object(self):
        # Select the non_linear type object to use
        if(self.non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(self.non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(self.non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(self.non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    

        return non_linear_object()

    def forward(self, x):

        # The first conv layers
        conv_layers_1_outs = []
        for cl in self.conv_layers_1:
            conv_layers_1_outs.append(cl(x))
        x = torch.cat(conv_layers_1_outs, dim=1)

        # Max pool!!!
        x = self.max_pooling_layer_1(x)
        x = self.activation_1(x)

        # The second conv layers
        conv_layers_2_outs = []
        for cl in self.conv_layers_2:
            conv_layers_2_outs.append(cl(x))
        x = torch.cat(conv_layers_2_outs, dim=1)

        # Max pool!!!
        x = self.max_pooling_layer_2(x)

        # Do the final activation
        x = self.activation_2(x)

        return x





def create_observation_model(model_name, model_parameters):

    model_type = model_parameters[model_name]["type"]

    if(model_type == "VectorObservationEncoder"):
        parameters = model_parameters[model_name]
        return VectorObservationEncoder(parameters)
    elif(model_type == "ImageObservationEncoder"):
        parameters = model_parameters[model_name]
        return ImageObservationEncoder(parameters)
    elif(model_type == "ImageObservationWithPoolingEncoder"):
        parameters = model_parameters[model_name]
        return ImageObservationWithPoolingEncoder(parameters)
    elif(model_type == "ResnetObservationEncoder"):
        parameters = model_parameters[model_name]
        return ResnetObservationEncoder(parameters)
    elif(model_type == "House3DObservationEncoder"):
        parameters = model_parameters[model_name]
        return House3DObservationEncoder(parameters)

    else:
        print("Unknown observation_encoder type \"{}\"".format(model_type))
        exit()


