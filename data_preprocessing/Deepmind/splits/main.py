import torch
from tqdm import tqdm 
import random


# the total number of sequences that are in the set of data
TOTAL_NUM_SEQUENCES = 1000

# The save dir location
save_dir = "../../../data/deep_mind_localization/data/100s/"

# Select how many sequences to use for each split.  Dont exceed how many sequences we actually have
number_of_training_seqs = 800
number_of_validation_seqs = 200
assert((number_of_training_seqs + number_of_validation_seqs) <= TOTAL_NUM_SEQUENCES)


for maze in ["nav01", "nav02", "nav03"]:

	# List out all the sequences
	all_sequences = [i for i in range(TOTAL_NUM_SEQUENCES)]

	# Shuffle the list so we can get random subsets
	random.shuffle(all_sequences)

	# Split
	training_sequences = all_sequences[0:number_of_training_seqs]
	validation_sequences = all_sequences[number_of_training_seqs:(number_of_training_seqs+number_of_validation_seqs)]

	# Create the save dict
	save_dict = dict()
	save_dict["training"] = training_sequences
	save_dict["validation"] = validation_sequences

	# None means use all the sequences
	save_dict["evaluation"] = None 

	# Save
	save_file = "{}/{}_splits.pt".format(save_dir, maze)
	torch.save(save_dict, save_file)
