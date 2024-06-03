# Standard Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import os 

# Pytorch Imports
import torch

# Project Imports
from evaluations.full_sequence_evaluations.full_sequence_evaluation_base import *


class BunchOFiles(animation.FileMovieWriter):
    '''
        Taken from:
            https://stackoverflow.com/questions/41230286/matplotlib-animation-write-to-png-files-without-third-party-module
    '''
    supported_formats = ['png', 'jpeg', 'bmp', 'svg', 'pdf']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, fig, outfile, dpi=None, frame_prefix=None):
        super().setup(fig, outfile, dpi, frame_prefix)

        # Set the temp prefix to be the file without the file format
        filename, file_extension = os.path.splitext(outfile)
        self.temp_prefix = filename + "_"

        # Change the file format to be a little nicer
        self.fname_format_str = '%s%%03d.%s'

    def finish(self):
        # Dont do any cleanup, we dont want to delete out files
        pass


class FullSequenceEvaluationVideoBase(FullSequenceEvaluationBase):
    def __init__(self, experiment, problem, model, save_dir, device, seed=0):
        super().__init__(experiment, problem, model, save_dir, device, seed)

    def render_sequence(self, render_index, data, output_dicts):

        # Get the sequence index, if one doesnt exist then just use the render index
        sequence_index = render_index
        if("dataset_index" in data):
            sequence_index = data["dataset_index"][0]

        # Extract some stats
        subsequence_length = self.evaluation_dataset.get_subsequence_length() 
        
        self.new_rendering()

        if(self.manually_manage_figure() == False):

            # For now we want just 1 col.  Maybe we can make this prettier later
            rows, cols = self.get_rows_and_cols()
            
            # Make the figure
            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(6*cols, 3*rows))

            # fig, axes = plt.subplots(rows, cols, sharex=False)
            axes = axes.reshape(-1,)

            # Nothing to init
            def init():
                pass

            # The main loop that will be used when rendering the frames
            def update(frame_number):

                ax_list = fig.axes

                # Clear all the axes
                for ax in fig.axes:
                    ax.clear()


                # # Clear all the axes
                # for i in range(axes.shape[0]):
                #     axes[i].clear()

                # Render this frame
                self.render_frame(frame_number, data, output_dicts, axes)

                fig.tight_layout()

            # create the animation
            anim = animation.FuncAnimation(fig, update, init_func=init, frames=subsequence_length, blit=False, interval=1000)

        else:
            fig = self.get_figure()

            def init():
                return self.init_rendering(data, output_dicts, None)

            # The main loop that will be used when rendering the frames
            def update(frame_number):

                # Render this frame
                return self.render_frame(frame_number, data, output_dicts, None)

            # create the animation
            anim = animation.FuncAnimation(fig, update, init_func=init, frames=subsequence_length, blit=True, interval=1000)

        # plt.show()


        ##############################################################################################################
        # Save the individual frames
        ##############################################################################################################
        single_frames_save_dir = "{}/detailed_rendering_{:03d}".format(self.save_dir, sequence_index)
        if(not os.path.exists(single_frames_save_dir)):
            os.makedirs(single_frames_save_dir)

        save_filepath = "{}/frames.png".format(single_frames_save_dir)
        # anim.save(save_filepath, writer="imagemagick")
        anim.save(save_filepath, writer=BunchOFiles())


        ##############################################################################################################
        # Save the Gif
        ##############################################################################################################
        fps = 5
        save_filepath = "{}/detailed_rendering_{:03d}.gif".format(self.save_dir, sequence_index)
        writergif = animation.PillowWriter(fps=fps) 
        anim.save(save_filepath, writer=writergif)


    def manually_manage_figure(self):
        return False

    def get_figure(self):
        raise NotImplemented

    def extract_if_present(self, key, dict_to_extract_from):
        if(key in dict_to_extract_from):
            return dict_to_extract_from[key]
        else:
            return None


    def new_rendering(self):
        return