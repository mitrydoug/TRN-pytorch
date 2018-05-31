import torch
import torch.nn as nn
import pdb

class LSTMModule(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_frames, img_feature_dim):
		self.lstm = nn.LSTM(input_size, hidden_size)
		self.h0 = torch.randn(1, batch_size, hidden_size)
		self.c0 = torch.randn(1, batch_size, hidden_size)
			
	def forward(self, x):
		pdb.set_trace()
		# Expects data to be of shape (batch_size, num_frames*self.img_feature_dim)
		batch_size, _ = x.size()
		# Reshape to (batch_size, img_feature_dim, num_frames)
		x = x.view(batch_size, self.num_frames, self.img_feature_dim)
		# Reshape to (num_frames, batch_size, img_feature_dim), as expected by pytorch
		x = x.permute(2, 0, 1)
		# Discard intermediate hidden states
		_, last_hidden = self.lstm(x)
		# Output is of size (batch_size, hidden_size)
		return last_hidden[0]

