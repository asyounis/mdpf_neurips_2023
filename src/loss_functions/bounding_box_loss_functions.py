
# Standard Imports
import torch
import torch.nn.functional as F

# Project Imports
from loss_functions.loss_function_base import *


class BoundingBoxIoU(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)


    def compute_loss(self, output_dict, states):
        
        # Unpack the output dict to get what we need for the loss
        particles = output_dict["particles"]
        particle_weights = output_dict["particle_weights"]

        # Tile the states so that it has the same dim as the particles
        tiled_states = torch.tile(states.unsqueeze(1), [1, particles.shape[1], 1])

        #########################################################
        # Compute the intersection of the boxes
        #########################################################

        # First need to compute the left and right X values
        gt_ulx, gt_uly, gt_lrx, gt_lry = self.convert_from_center_and_size_to_corners_representation(tiled_states)
        predicted_ulx, predicted_uly, predicted_lrx, predicted_lry = self.convert_from_center_and_size_to_corners_representation(particles)

        # Compute the corners for the intersection 
        x_a = torch.maximum(gt_ulx, predicted_ulx)
        y_a = torch.maximum(gt_uly, predicted_uly)
        x_b = torch.minimum(gt_lrx, predicted_lrx)
        y_b = torch.minimum(gt_lry, predicted_lry)


        # Compute the width and height (aka the corner differences)
        diff_x = x_b-x_a
        diff_y = y_b-y_a

        # Make sure that the values are positive
        diff_x = F.relu(diff_x)
        diff_y = F.relu(diff_y)

        # compute the intersection area
        intersection_area = diff_x * diff_y

        #########################################################
        # Compute the area of the boxes
        #########################################################
        gt_area = tiled_states[..., -1] * tiled_states[..., -2]
        predicted_area = particles[..., 2] * particles[..., 3]


        #########################################################
        # Compute the intersection over union (IoU)
        #########################################################
        iou = intersection_area / (gt_area + predicted_area - intersection_area)

        assert(torch.sum(iou > 1.00001) == 0)
        assert(torch.sum(iou < 0.0) == 0)


        #########################################################
        # Compute the final loss
        #########################################################
        loss = iou * particle_weights
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        loss = -loss

        return loss



    def convert_from_center_and_size_to_corners_representation(self, boxes):
        ''' 
            Convert the bounding boxes from [center_x, center_y, width, height] to [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        '''

        ulx = boxes[..., 0] - (boxes[..., 2] / 2.0) 
        uly = boxes[..., 1] - (boxes[..., 3] / 2.0) 
        lrx = boxes[..., 0] + (boxes[..., 2] / 2.0) 
        lry = boxes[..., 1] + (boxes[..., 3] / 2.0) 

        return ulx, uly, lrx, lry




class BoundingBoxIoUSongleSolution(LossFunctionBase):
    def __init__(self, loss_params, model):
        super().__init__(model)


    def compute_loss(self, output_dict, states):
        
        # Unpack the output dict to get what we need for the loss
        particles = output_dict["particles"]
        particle_weights = output_dict["particle_weights"]

        # Compute the mean particle
        mean_particle = torch.sum(particles * particle_weights.unsqueeze(-1), dim=1)

        #########################################################
        # Compute the intersection of the boxes
        #########################################################

        # First need to compute the left and right X values
        gt_ulx, gt_uly, gt_lrx, gt_lry = self.convert_from_center_and_size_to_corners_representation(states)
        predicted_ulx, predicted_uly, predicted_lrx, predicted_lry = self.convert_from_center_and_size_to_corners_representation(mean_particle)

        # Compute the corners for the intersection 
        x_a = torch.maximum(gt_ulx, predicted_ulx)
        y_a = torch.maximum(gt_uly, predicted_uly)
        x_b = torch.minimum(gt_lrx, predicted_lrx)
        y_b = torch.minimum(gt_lry, predicted_lry)


        # Compute the width and height (aka the corner differences)
        diff_x = x_b-x_a
        diff_y = y_b-y_a

        # Make sure that the values are positive
        diff_x = F.relu(diff_x)
        diff_y = F.relu(diff_y)

        # compute the intersection area
        intersection_area = diff_x * diff_y

        #########################################################
        # Compute the area of the boxes
        #########################################################
        gt_area = states[..., -1] * states[..., -2]
        predicted_area = mean_particle[..., 2] * mean_particle[..., 3]


        #########################################################
        # Compute the intersection over union (IoU)
        #########################################################
        iou = intersection_area / (gt_area + predicted_area - intersection_area)

        assert(torch.sum(iou > 1.00001) == 0)
        assert(torch.sum(iou < 0.0) == 0)


        #########################################################
        # Compute the final loss
        #########################################################
        loss = iou
        loss = torch.mean(loss)
        loss = -loss

        return loss



    def convert_from_center_and_size_to_corners_representation(self, boxes):
        ''' 
            Convert the bounding boxes from [center_x, center_y, width, height] to [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        '''

        ulx = boxes[..., 0] - (boxes[..., 2] / 2.0) 
        uly = boxes[..., 1] - (boxes[..., 3] / 2.0) 
        lrx = boxes[..., 0] + (boxes[..., 2] / 2.0) 
        lry = boxes[..., 1] + (boxes[..., 3] / 2.0) 

        return ulx, uly, lrx, lry