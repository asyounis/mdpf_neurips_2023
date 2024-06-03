# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from multiprocessing import Pool

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, start_offset=20, max_lr_change=3):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param start_offset: minimum number of steps before early stopping is 
               considered
        """
        self.patience = patience
        self.min_delta = min_delta
        self.start_offset = start_offset
        self.max_lr_change = max_lr_change

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.lr_change_counter = 0

    def __call__(self, val_loss):

        if(self.start_offset > 0):
            self.start_offset -= 1
            return

        if(self.best_loss == None):
            self.best_loss = val_loss

        elif((self.best_loss - val_loss) > self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            
        elif ((self.best_loss - val_loss) < self.min_delta):
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if (self.counter >= self.patience):
                print('INFO: Early stopping')
                self.early_stop = True

    def lr_changed(self):
        self.lr_change_counter += 1
        if(self.lr_change_counter <= self.max_lr_change):
            self.counter = 0




class DataPlotter:
    def __init__(self, title, x_axis_label, y_axis_label, save_dir, filename, moving_average_length=1, plot_modulo=1000, save_raw_data=False):

        # Save all the info we need to keep around
        self.title = title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.save_dir = save_dir
        self.filename = filename
        self.moving_average_length = moving_average_length
        self.plot_modulo = plot_modulo
        self.save_raw_data = save_raw_data

        # The data structs that we will use to keep track of the data
        self.raw_values = []
        self.averaged_values = []
        self.moving_aveage_buffer = []

        # Make sure the directory we are going to save into exists
        if not os.path.exists("{}/".format(self.save_dir)):
            os.makedirs("{}//".format(self.save_dir))



    def make_copy(self):
        dp_copy = DataPlotter(title=self.title, x_axis_label=self.x_axis_label, y_axis_label=self.y_axis_label, save_dir=self.save_dir, filename=self.filename, moving_average_length=self.moving_average_length, plot_modulo=self.plot_modulo, save_raw_data=self.save_raw_data)
        dp_copy.raw_values = copy.deepcopy(self.raw_values)
        dp_copy.averaged_values = copy.deepcopy(self.averaged_values)
        dp_copy.moving_aveage_buffer = copy.deepcopy(self.moving_aveage_buffer)

        return dp_copy


    def add_value(self, value):

        # Keep track of the raw value 
        self.raw_values.append(value)

        # Add the values to the average
        self.add_value_to_average(value)

        # Check if we should plot on this iteration
        if((len(self.averaged_values) % self.plot_modulo) == 0):
            self.plot_and_save()

    def add_value_to_average(self, value):

        # Add the value to the averaging buffer and make sure that the averaging
        # buffer size is constant
        self.moving_aveage_buffer.append(value)

        # If we dont have enough samples then we dont have enough to add to average 
        if(len(self.moving_aveage_buffer) < self.moving_average_length):
            return

        # If we have too many samples then we need to pop some until we have the right number of samples
        while(len(self.moving_aveage_buffer) > self.moving_average_length):
            self.moving_aveage_buffer.pop(0)

        # compute and add the average value
        avg = sum(self.moving_aveage_buffer) / float(self.moving_average_length)
        self.averaged_values.append(avg)

    def plot_and_save(self):

        # check if there is something to plot 
        if((len(self.raw_values) == 0) or (len(self.averaged_values) == 0)):
            return

        self.plot_and_save_helper()

        # dp_copy = self.make_copy()

        # with Pool(1) as pool:
        #     pool.map(DataPlotter.plot_and_save_helper, [dp_copy])


    def plot_and_save_helper(dp_obj):


        # check if there is something to plot 
        if((len(dp_obj.raw_values) == 0) or (len(dp_obj.averaged_values) == 0)):
            return

        # Plot the losses.  This overrides the previous plots
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(12, 6))
        ax1 = axes[0,0]

        # Plot the Training loss
        ax1.plot(dp_obj.averaged_values, marker="o", color="red")

        # Add labels every so often so we can read the data
        for i in range(0, len(dp_obj.averaged_values), 10):
            ax1.text(i, dp_obj.averaged_values[i], "{:.2f}".format(dp_obj.averaged_values[i]))

        # Plot the trend line if we have enough data
        if(len(dp_obj.averaged_values) > 5):
            x = np.arange(0, len(dp_obj.averaged_values), 1)
            y = np.asarray(dp_obj.averaged_values)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax1.plot(x,p(x),"b--")


        ax1.set_xlabel(dp_obj.x_axis_label)
        ax1.set_ylabel(dp_obj.y_axis_label)
        ax1.set_title(dp_obj.title)
        # ax1.legend()
        ax1.get_yaxis().get_major_formatter().set_scientific(False)


        # Compute Stats to put in the table
        raw_values_array = np.asarray(dp_obj.raw_values)
        mean = np.mean(raw_values_array)
        std = np.std(raw_values_array)
        median = np.median(raw_values_array)
        max_value = np.max(raw_values_array)
        min_value = np.min(raw_values_array)
        number_of_maxs = np.sum(np.abs(raw_values_array - max_value) < 1e-3)
        percent_of_maxs = float(number_of_maxs) / raw_values_array.shape[0]

        q75, q25 = np.percentile(raw_values_array, [75.0 ,25.0])
        iqr = q75 - q25


        columns = ["Stat", "Value"]
        cell_text = list()
        cell_text.append(["Mean", mean])
        cell_text.append(["STD", std])
        cell_text.append(["Median", median])
        cell_text.append(["IQR", iqr])
        cell_text.append(["Max Value", max_value])
        cell_text.append(["Min Value", min_value])
        cell_text.append(["Num Max Values", number_of_maxs])
        cell_text.append(["Percent Max Values", percent_of_maxs])
        cell_text.append(["Total Steps", raw_values_array.shape[0]])

        # ax1.table(cellText=cell_text, colLabels=columns, loc='bottom', bbox=(1.1, .2, 0.5, 0.5))
        ax1.table(cellText=cell_text, colLabels=columns, bbox=(1.1, .2, 0.5, 0.5))

        fig.tight_layout()


        # Save into the save dir, overriding any previously saved plots
        plt.savefig("{}/{}".format(dp_obj.save_dir, dp_obj.filename))

        # Close the figure when we are done to stop matplotlub from complaining
        plt.close('all')


        # If we should save the raw data then lets save it
        if(dp_obj.save_raw_data):

            # Convert to a torch tensor for saving
            raw_values_torch = torch.FloatTensor(dp_obj.raw_values)

            # Save it
            torch.save(raw_values_torch, "{}/{}.pt".format(dp_obj.save_dir, dp_obj.filename))




class LossShaper:
    def __init__(self, loss_shaper_params):

        # The parameters
        self.loss_shaper_params = loss_shaper_params
        self.starting_coeff = self.loss_shaper_params["starting_coeff"]
        self.ending_coeff = self.loss_shaper_params["ending_coeff"]
        self.change_over_steps = self.loss_shaper_params["change_over_steps"]

        # The step counter
        self.step_counter = 0
        self.current_coeff = self.starting_coeff

    def shape_loss(self, loss, seq_idx):

        # Scale the loss
        return loss * (self.current_coeff**(seq_idx))

    def step(self):
        self.step_counter += 1

        # Compute the current coeff
        alpha = float(self.step_counter) / float(self.change_over_steps) 
        alpha = min(alpha, 1.0)
        self.current_coeff = self.starting_coeff*(1-alpha) + self.ending_coeff*alpha


    def reset(self):
        self.step_counter = 0
        self.current_coeff = self.starting_coeff






class CustomReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):

        self.start_epoch = kwargs["start_epoch"]
        del kwargs["start_epoch"]

        super(CustomReduceLROnPlateau, self).__init__(*args, **kwargs)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch


        if(self.last_epoch < self.start_epoch):
            return False



        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        did_reduce_lr = False

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            did_reduce_lr = True

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        return did_reduce_lr


class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class CustomStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, *args, **kwargs):

        self.start_epoch = kwargs["start_epoch"]
        del kwargs["start_epoch"]

        super(CustomStepLR, self).__init__(*args, **kwargs)


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if(self._step_count < self.start_epoch):
            return False


        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values, did_reduce_lr = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values, did_reduce_lr = self._get_closed_form_lr()
                else:
                    values, did_reduce_lr = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        return did_reduce_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups], False
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups], True

    def _get_closed_form_lr(self):

        did_reduce_lr = False
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            did_reduce_lr = True

        return [base_lr * self.gamma ** (self.last_epoch // self.step_size) for base_lr in self.base_lrs], did_reduce_lr
