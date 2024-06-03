
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
from trainers.optimizers.rmsprop import *

class Trainer():
    def __init__(self, params, main_model, problem, save_dir, device):
            
        # Keep track of some of the important things
        self.main_model = main_model
        self.save_dir = save_dir
        self.device = device
        self.problem = problem

        # Create a directory to save the models to
        self.model_save_dir = self.save_dir + "/models/"
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Parse the command arguments
        self.training_params = params["training_params"]

        # Parse the parameters
        self.epochs = self.training_params["epochs"]
        self.early_stopping_patience = self.training_params["early_stopping_patience"]
        self.early_stopping_start_offset = self.training_params["early_stopping_start_offset"]
        self.training_batch_size = self.training_params["training_batch_size"]
        self.validation_batch_size = self.training_params["validation_batch_size"]

        if("num_cpu_cores_for_dataloader" in self.training_params):
            self.num_cpu_cores_for_dataloader = self.training_params["num_cpu_cores_for_dataloader"]
        else:
            self.num_cpu_cores_for_dataloader = 4

        if("accumulate_gradients_counter" in self.training_params):
            self.accumulate_gradients_counter = self.training_params["accumulate_gradients_counter"]
        else:
            self.accumulate_gradients_counter = 1

        if("gradient_value_for_skip" in self.training_params):
            self.gradient_value_for_skip = self.training_params["gradient_value_for_skip"]
        else:
            self.gradient_value_for_skip = None

        if("gradient_clip_value" in self.training_params):
            self.gradient_clip_value = self.training_params["gradient_clip_value"]
        else:
            self.gradient_clip_value = None

        if("gradient_clip_params" in self.training_params):
            assert(self.gradient_clip_value is None)
            assert(self.gradient_value_for_skip is None)

            self.gradient_clip_params = self.training_params["gradient_clip_params"]
        else:
            self.gradient_clip_params = None


        # Extract the optimizer type
        self.optimizer_type = self.training_params["optimizer_type"]

        # Extract specific optimizer types if it is present
        if("specific_optimizer_type" in self.training_params):
            self.specific_optimizer_type = self.training_params["specific_optimizer_type"]
        else:
            # We dont have any specific optimizers to use
            self.specific_optimizer_type = dict()

        # Extract the lr_scheduler parameters
        self.lr_scheduler_params = self.training_params["lr_scheduler_params"]

        # If we have weight decay then we want to load it
        if "weight_decay" in self.training_params:
            self.weight_decay = self.training_params["weight_decay"]
        else:
            self.weight_decay = 0.0

        # If we have optimizer momentum then we want to load it
        if "optimizer_momentum" in self.training_params:
            self.optimizer_momentum = self.training_params["optimizer_momentum"]
        else:
            self.optimizer_momentum = 0.0


        # If we have optimizer momentum then we want to load it
        if "optimizer_dampening" in self.training_params:
            self.optimizer_dampening = self.training_params["optimizer_dampening"]
        else:
            self.optimizer_dampening = 0.0


        # Load the RMSProp parameters
        if((self.optimizer_type == "RMSProp") or (self.optimizer_type == "RMSPropAli")):
            if ("rmsprop_alpha" in self.training_params):
                self.rmsprop_alpha = self.training_params["rmsprop_alpha"]                
            else:
                self.rmsprop_alpha = 0.99

            if ("rmsprop_eps" in self.training_params):
                self.rmsprop_eps = self.training_params["rmsprop_eps"]
            else:
                self.rmsprop_eps = 1e-8


        # Check if we should save the training gradient information
        if("save_raw_gradient_information" in self.training_params):
            self.save_raw_gradient_information = self.training_params["save_raw_gradient_information"]
        else:
            self.save_raw_gradient_information = False

        # If we should save all the intermediate models or just the best ones.  Be default we save only the best ones
        if("save_intermediate_models" in self.training_params):
            self.save_intermediate_models = self.training_params["save_intermediate_models"]
        else:
            self.save_intermediate_models = False

      # If we should save all the intermediate models or just the best ones.  Be default we save only the best ones
        if("regularization_params" in self.training_params):
            self.regularization_params = self.training_params["regularization_params"]
        else:
            self.regularization_params = None



        # Get the datasets
        self.training_dataset = self.problem.get_training_dataset()
        self.validation_dataset = self.problem.get_validation_dataset()

        # Create the dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_dataset, batch_size=self.training_batch_size, shuffle=True, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True, persistent_workers=True, collate_fn=self.training_dataset.get_collate_fn())
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.validation_dataset, batch_size=self.validation_batch_size, shuffle=False, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True, persistent_workers=True, collate_fn=self.validation_dataset.get_collate_fn())

        # Create the early stopping thingy
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, start_offset=self.early_stopping_start_offset)

        # The actual objects we will be using for training
        self.models = dict()
        self.optimizers = []
        self.lr_schedulers = []
        self.models_to_optimize = []


        # The Plotters that we will be using
        self.all_data_plotters = dict()
        self.all_data_plotters["training_loss_plotter"] = DataPlotter("Training Loss", "steps", "NLL Loss", save_dir, "training_step_loss_curves.png")
        self.all_data_plotters["validation_loss_plotter"] = DataPlotter("Validation Loss", "steps", "NLL Loss", save_dir, "validation_step_loss_curves.png")
        
        # Keep track of the training and validation losses for plotting purposes
        self.training_losses = []
        self.validation_losses = []
        self.changed_lr_epochs = []
        self.gradient_norms = dict()

        # Number of skipped Gradients
        self.total_number_of_skipped_gradients = 0

        # The large gradient counter
        self.large_gradient_output_save_counter = 0

    def get_training_params(self) :
        return self.training_params

    def get_training_type(self):
        raise NotImplemented

    def do_freeze_rnn_batchnorm_layers(self):
        # By default freeze nothing
        return False


    def create_lr_scheduler(self, optimizer, lr_scheduler_params):

        scheduler_type = lr_scheduler_params["scheduler_type"]
        if(scheduler_type == "ReduceLROnPlateau"):

            # Extract the parameters of this scheduler
            threshold = lr_scheduler_params["threshold"]
            factor = lr_scheduler_params["factor"]
            patience = lr_scheduler_params["patience"]
            cooldown = lr_scheduler_params["cooldown"]
            min_lr = lr_scheduler_params["min_lr"]
            verbose = lr_scheduler_params["verbose"]

            if("start_epoch" in lr_scheduler_params):
                start_epoch = lr_scheduler_params["start_epoch"]
            else:
                start_epoch = 0

            # Create the scheduler
            return CustomReduceLROnPlateau(optimizer, 'min', start_epoch=start_epoch, threshold=threshold, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr, verbose=verbose)
        
        elif(scheduler_type == "StepLR"):

            # Extract the parameters of this scheduler
            step_size = lr_scheduler_params["step_size"]
            gamma = lr_scheduler_params["gamma"]
            verbose = lr_scheduler_params["verbose"]

            if("start_epoch" in lr_scheduler_params):
                start_epoch = lr_scheduler_params["start_epoch"]
            else:
                start_epoch = 0

            # Create the scheduler
            return CustomStepLR(optimizer, step_size=step_size, gamma=gamma, verbose=verbose, start_epoch=start_epoch)

        elif(scheduler_type == "OneCycleLR"):

            # Havent implemented the custom one yet
            assert(False)

            # Extract the parameters of this scheduler
            max_lr = lr_scheduler_params["max_lr"]
            verbose = lr_scheduler_params["verbose"]

            # Create the scheduler
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=self.epochs, steps_per_epoch=len(self.train_loader), verbose=verbose)

        else:
            print("Unknown lr_scheduler")
            exit()

    def add_and_plot_losses(self, training_loss, validation_loss, did_change_lr, epoch):

        # Keep track of the losses
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)

        # Keep track of every time we changed the LR
        if(did_change_lr):
            self.changed_lr_epochs.append(epoch)

        # Plot the losses.  This overrides the previous plots
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)


        step = int(len(self.training_losses) / 10) + 1  

        # Plot the Training loss
        ax1.plot(self.training_losses, marker="o", label="Training", color="red")
        for i in range(0,len(self.training_losses),step):
            ax1.text(i, self.training_losses[i], "{:.2f}".format(self.training_losses[i]))
        for lr_change_epoch in self.changed_lr_epochs:
            ax1.axvline(x=lr_change_epoch, color="blue")
            
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Average Loss For Epoch")
        ax1.legend()
        ax1.get_yaxis().get_major_formatter().set_scientific(False)

        # Plot the Validation loss
        ax2.plot(self.validation_losses, marker="o", label="Validation", color="green")
        for i in range(0,len(self.validation_losses),step):
            ax2.text(i, self.validation_losses[i], "{:.2f}".format(self.validation_losses[i]))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Average Loss For Epoch")
        ax2.legend()
        ax2.get_yaxis().get_major_formatter().set_scientific(False)


        # Save into the save dir, overriding any previously saved loss plots
        plt.savefig("{}/loss_curves.png".format(self.save_dir))

        # Close the figure when we are done to stop matplotlub from complaining
        plt.close('all')


    def add_loss_function(self, loss_function):
        self.loss_function = loss_function

    def add_model(self, model, model_name, do_grad_clipping=False): 
        self.models[model_name] = (model, do_grad_clipping)

        # Create gradient plotters
        plotter_name = "{}_gradient_plotter".format(model_name)
        self.all_data_plotters[plotter_name] = DataPlotter("Gradient L2 Norm of {}".format(model_name), "steps", "Gradient L2 Norm", "{}/gradients/".format(self.save_dir), "{}.png".format(model_name), save_raw_data=self.save_raw_gradient_information)

        # Create the l2 norm weight
        plotter_name = "{}_l2_norm_plotter".format(model_name)
        self.all_data_plotters[plotter_name] = DataPlotter("Weight L2 Norm of {}".format(model_name), "steps", "Weight L2 Norm", "{}/weights_l2_norm/".format(self.save_dir), "{}.png".format(model_name))

    def add_optimizer_and_lr_scheduler(self, models, model_names, learning_rate):

        # We only support 1 model at a time for now
        assert(len(models) == 1)
        assert(len(model_names) == 1)

        # Gather all the params to add to the optimizer
        params_for_optimizer = list()
        for model in models:
            if(model is not None):
                params_for_optimizer += list(model.parameters())    

        if((learning_rate == "Freeze") or (len(params_for_optimizer) == 0)):

            # If the learning rate is frozen then we want to mark the params as not needing a gradient 
            # and also mark them as being in eval mode so that things like batchnorm are also frozen
            for model in models:
                if(model is not None):
                    for params in model.parameters():
                        params.requires_grad = False
                    model.eval()

        else:

            # Get which optimizer to use.  The default one or a different one
            optimizer_to_use = self.optimizer_type
            if(model_names[0] in self.specific_optimizer_type):
                optimizer_to_use = self.specific_optimizer_type[model_names[0]]

            if(optimizer_to_use == "Adam"):
                optimizer = torch.optim.Adam(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay)
            elif(optimizer_to_use == "AdamW"):
                optimizer = torch.optim.AdamW(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay)
            elif(optimizer_to_use == "NAdam"):
                optimizer = torch.optim.NAdam(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay)
            elif(optimizer_to_use == "RMSProp"):
                optimizer = torch.optim.RMSprop(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay, momentum=self.optimizer_momentum, eps=self.rmsprop_eps, alpha=self.rmsprop_alpha)
            elif(optimizer_to_use == "RMSPropAli"):
                optimizer = RMSpropCustom(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay, momentum=self.optimizer_momentum, eps=self.rmsprop_eps, alpha=self.rmsprop_alpha)
            elif(optimizer_to_use == "SGD"):
                optimizer = torch.optim.SGD(params_for_optimizer,lr=learning_rate, weight_decay=self.weight_decay, momentum=self.optimizer_momentum, dampening=self.optimizer_dampening)
            else:
                print("Unknown optimizer being used: ", optimizer_to_use)
                exit()

            self.optimizers.append(optimizer)

            # Create the LR scheduler with the default params
            lr_scheduler = self.create_lr_scheduler(optimizer,self.lr_scheduler_params)
            self.add_lr_scheduler(lr_scheduler)

            # Add all the names passed in as 
            self.models_to_optimize.extend(model_names)

    def add_model_for_training(self, model_name):
        self.models_to_optimize.append(model_name)

    def add_lr_scheduler(self, lr_scheduler):
        self.lr_schedulers.append(lr_scheduler)

    def do_training_epoch(self, epoch):

        # Set the models to training mode if we are training that model
        for model_name in self.models.keys():
            self.models[model_name][0].eval()
            # print("Eval", model_name)


        if(len(self.models_to_optimize) != 0):
            self.main_model.train()

        for model_name in self.models_to_optimize:
            self.models[model_name][0].train()

            if(epoch == 0):
                print("train", model_name)

        for model_name in self.models.keys():
            if((model_name != "full_dpf_model") and (model_name not in self.models_to_optimize)):
                self.models[model_name][0].eval()
                
                if(epoch == 0):
                    print("Eval", model_name)

        # Freeze the batchnorm layers if we are supposed to
        if(self.do_freeze_rnn_batchnorm_layers()):
            self.main_model.freeze_rnn_batchnorm_layers()


        # Keep track of the average loss over this epoch
        average_loss = 0

        # Prepare for the optimization
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        
        for model_name in self.models.keys():
            self.models[model_name][0].zero_grad()


        # The number of losses used when taking steps
        number_of_losses_to_use_for_average_loss = 0

        output_file = open(self.save_dir + "/error_seq.txt", "w")
        output_file.write("\n")
        output_file.close()


        # Go through all the data once
        t = tqdm(iter(self.train_loader), leave=False, total=len(self.train_loader))
        for step, data in enumerate(t):
            
            # Do the forward pass over the data
            loss, batch_size = self.do_forward_pass(data, self.training_dataset, epoch)

            # If the loss is not valid then move on
            if(loss is None):
                continue

            if(self.regularization_params is not None):
                reg_type = self.regularization_params["type"]
                reg_lambda = self.regularization_params["lambda"]

                if(reg_type == "L1"):

                    all_params = []
                    all_params.extend([x.view(-1) for x in self.models["proposal_model"][0].parameters()])
                    all_params.extend([x.view(-1) for x in self.models["action_encoder_model"][0].parameters()])
                    all_params.extend([x.view(-1) for x in self.models["particle_encoder_for_particles_model"][0].parameters()])
                    all_params = torch.cat(all_params)
                    

                    # all_params = torch.cat([x.view(-1) for x in self.main_model.parameters()])
                    reg_value = torch.norm(all_params, 1)
                    reg_value = reg_value * reg_lambda
                elif(reg_type == "L2"):

                    all_params = []
                    all_params.extend([x.view(-1) for x in self.models["proposal_model"][0].parameters()])
                    # all_params.extend([x.view(-1) for x in self.models["action_encoder_model"][0].parameters()])
                    # all_params.extend([x.view(-1) for x in self.models["particle_encoder_for_particles_model"][0].parameters()])
                    all_params = torch.cat(all_params)
                    reg_value = torch.norm(all_params, 2)
                    reg_value = reg_value * reg_lambda

                else:
                    assert(False)

                loss_with_reg = loss + reg_value 

                loss_with_reg.backward(retain_graph=True)
            else:


                # Compute the gradient
                # loss.backward()
                loss.backward(retain_graph=True)

            if((((step+1) % self.accumulate_gradients_counter) == 0) or ((step+1) == len(self.train_loader))):

                norm_type = 2


                # norm = [torch.norm(p.grad.detach(), norm_type) for p in self.main_model.parameters() if p.grad is not None]
                # gradient_norm = torch.norm(torch.stack(norm) , norm_type)
                # if(gradient_norm.item() > 400):
                #     exit()



                # If we have a gradient value that we should use for skipping then we should skip if we get that gradient value
                if(self.gradient_value_for_skip is not None):

                    # # Compute the overall gradient value for the model
                    # norms = []
                    # for model_name in self.models.keys():
                    #     norm = [torch.norm(p.grad.detach(), norm_type) for p in self.models[model_name][0].parameters() if p.grad is not None]
                    #     norms.extend(norm)
                    # norms = torch.stack(norms)
                    # overall_gradient_norm = torch.norm(norms , norm_type)

                    # Compute the gradient norm for all the values
                    norm = [torch.norm(p.grad.detach(), norm_type) for p in self.main_model.parameters() if p.grad is not None]
                    gradient_norm = torch.norm(torch.stack(norm) , norm_type)
                    



                    # If the value is larger than the max then we need to skip this gradient step
                    if(gradient_norm.item() >= self.gradient_value_for_skip):

                        # Prepare for the next round of optimization
                        for optimizer in self.optimizers:
                            optimizer.zero_grad()
                        
                        # Manually zero out the gradients for all the models, not just the ones being optimized over
                        for model_name in self.models.keys():
                            self.models[model_name][0].zero_grad()

                        # Print so we know how many we are skipping
                        self.total_number_of_skipped_gradients += 1
                        print("")
                        print("Total Number of Skipped Gradients: {},    Step {},  Grad Value {}".format(self.total_number_of_skipped_gradients, step, gradient_norm.item()))

                        # Skip this round of optimization and move on 
                        continue    



                # if we have a gradient clipping value then do the clipping
                if(self.gradient_clip_value is not None):
                    total_norm = torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), self.gradient_clip_value)

                if(self.gradient_clip_params is not None):
                    for group_name in self.gradient_clip_params:
                        
                        # Extract the params
                        group_params = self.gradient_clip_params[group_name]
                        models_in_clipping_group = group_params["models"]
                        clip_value = group_params["clip_value"]

                        # Get all the parameters we want to clip
                        params_for_clipping = []
                        for model_name in models_in_clipping_group:
                            if(model_name in self.models):
                                params_for_clipping.extend(self.models[model_name][0].parameters())

                        # Do the clipping
                        total_norm = torch.nn.utils.clip_grad_norm_(params_for_clipping, clip_value)



                # Compute the norm of the gradients for plotting
                for model_name in self.models.keys():
                    norm = [torch.norm(p.grad.detach(), norm_type) for p in self.models[model_name][0].parameters() if p.grad is not None]
                    if(len(norm) != 0):
                        gradient_norm = torch.norm(torch.stack(norm) , norm_type)
                        plotter_name = "{}_gradient_plotter".format(model_name)
                        self.all_data_plotters[plotter_name].add_value(gradient_norm.item())

                # Take a step
                for optimizer in self.optimizers:
                    optimizer.step()

                # Update the learning rates if needed
                for lr_scheduler in self.lr_schedulers:

                    if(isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                        lr_scheduler.step()
                    else:
                        # Do nothing here, we update the learning rate scheduler after each epoch not each batch for other scheduler methods
                        pass

                # Prepare for the next round of optimization
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # Manually zero out the gradients for all the models, not just the ones being optimized over
                for model_name in self.models.keys():
                    self.models[model_name][0].zero_grad()

            # Add the loss for the batch so we can do step losses
            self.all_data_plotters["training_loss_plotter"].add_value(torch.mean(loss).cpu().item())

            # keep track of the average loss
            average_loss += loss.item() * batch_size
            number_of_losses_to_use_for_average_loss += batch_size

        # If we did not take any steps then WTF!!!!! 
        if(number_of_losses_to_use_for_average_loss == 0):
            assert(False)


        # Compute and return the average loss
        average_loss = average_loss / float(number_of_losses_to_use_for_average_loss)
        return average_loss

    def do_validation_epoch(self, epoch):

        # Dont need the gradients for the evaluation epochs
        with torch.no_grad():

            # Set the models to training mode
            for model_name in self.models.keys():
                self.models[model_name][0].eval()

            # Keep track of the average loss over this epoch
            average_loss = 0
            number_of_losses_to_use_for_average_loss = 0

            # Go through all the data once
            t = tqdm(iter(self.validation_loader), leave=False, total=len(self.validation_loader))
            for step, data in enumerate(t):
            # for step, data in enumerate(self.validation_loader):

                # Do the forward pass over the data
                loss, batch_size = self.do_forward_pass(data, self.validation_dataset, epoch)

                # If the loss is invalid then somethings is wrong
                if(loss is None):
                    # continue
                    print("Loss error")
                    exit()

                # keep track of the average loss
                average_loss += (loss.item() * batch_size)
                number_of_losses_to_use_for_average_loss += batch_size

                # Add the loss for the batch so we can do step losses
                self.all_data_plotters["validation_loss_plotter"].add_value(torch.mean(loss).cpu().item())

            # Compute and return the average loss
            average_loss = average_loss / float(number_of_losses_to_use_for_average_loss)
            return average_loss

    def train(self):

        # Save the untrained models for comparisons 
        for model_name in self.models.keys():
            model_save_name = "{}/{}_untrained.pt".format(self.model_save_dir, model_name)
            torch.save(self.models[model_name][0].state_dict(), model_save_name)

        # Keep track of the validation loss so we can save the best models
        best_validation_loss = None

        for epoch in tqdm(range(self.epochs)):
            
            # Do The training pass
            training_loss = self.do_training_epoch(epoch)

            # Do the validation pass
            validation_loss = self.do_validation_epoch(epoch)

            # Compute the l2 norm for all the models
            for model_name in self.models.keys():

                # Compute the l2 norm of the weights
                norm_type = 2
                total_norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in self.models[model_name][0].parameters()]) , norm_type)

                # Add it to the plotter
                plotter_name = "{}_l2_norm_plotter".format(model_name)
                self.all_data_plotters[plotter_name].add_value(total_norm.item())


            # Keep track of the best models so we can save them for later
            did_find_new_best = False 
            if((best_validation_loss is None) or (validation_loss < best_validation_loss)):
                best_validation_loss = validation_loss
                did_find_new_best = True


            did_change_lr = False

            # Update the learning rates
            for lr_scheduler in self.lr_schedulers:

                if(isinstance(lr_scheduler, CustomReduceLROnPlateau)):
                    lr_changed = lr_scheduler.step(validation_loss)
                elif(isinstance(lr_scheduler, CustomStepLR)):
                    lr_changed = lr_scheduler.step()
                elif(isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                    # Do nothing here.  We instead update the learning rate scheduler after each batch not here.
                    pass
                else:
                    print("NOOOOOO!! Unknown learning_rate_function")
                    exit()

                # If anything changed then flag it
                did_change_lr = (did_change_lr or lr_changed)

            # Plot the losses
            self.add_and_plot_losses(training_loss, validation_loss, did_change_lr, epoch)

            # Plot all the other plots at the end of the epoch
            for plotter_name in self.all_data_plotters.keys():
                self.all_data_plotters[plotter_name].plot_and_save()

            # If we changed the LR then tell the early stopping to reset
            if(did_change_lr):
                self.early_stopping.lr_changed()

            # Check for early stopping
            self.early_stopping(validation_loss)
            if(self.early_stopping.early_stop):
                break

            # Save the models for this epoch if we are told to save
            if(self.save_intermediate_models):
                for model_name in self.models.keys():
                    model_save_name = "{}/{}_{:04d}.pt".format(self.model_save_dir, model_name, epoch)
                    torch.save(self.models[model_name][0].state_dict(), model_save_name)

            # Save the best model
            if(did_find_new_best):
                for model_name in self.models.keys():
                    model_save_name = "{}/{}_best.pt".format(self.model_save_dir, model_name)
                    torch.save(self.models[model_name][0].state_dict(), model_save_name)







