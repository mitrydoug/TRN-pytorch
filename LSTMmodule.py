import torch
import torch.nn as nn

class LSTMModule(torch.nn.Module):
	def __init__(self, input_size, output_size, num_frames, img_feature_dim):
		self.lstm = nn.LSTM(input_size, output_size)
		self.h0 = 
		self.c0 = 
	def forward(self, x):
		# Expects data to be of shape (batch_size, num_frames*self.img_feature_dim)
		batch_size, _ = x.size()
		# Reshape to (batch_size, img_feature_dim, num_frames)
		x = x.view(batch_size, self.num_frames, self.img_feature_dim)
		# Reshape to (num_frames, batch_size, img_feature_dim), as expected by pytorch
		x = x.permute(2, 0, 1)
		# Discard intermediate hidden states
		_, last_hidden = self.lstm(x, self.hidden)
		# Output is of size (batch_size, output_size)
		return last_hidden

