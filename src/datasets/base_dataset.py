# Imports
import torch


class BaseDataset(torch.utils.data.Dataset):
	def __init__(self):
		self.disabled_outputs = set()

	def get_collate_fn(self):
		# None means that there is no collate function to return
		return None

	def disable_output(self, output_name):
		self.disabled_outputs.add(output_name)
