import torch
import torch.nn as nn
import pdb

class LSTMModule(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_frames, img_feature_dim):
		super().__init__()
		self.lstm = nn.GRU(img_feature_dim, hidden_size)
		self.num_frames = num_frames
		self.img_feature_dim = img_feature_dim
		self.h0 = torch.autograd.Variable(torch.randn(1, batch_size, hidden_size)).cuda()
		self.c0 = torch.autograd.Variable(torch.randn(1, batch_size, hidden_size)).cuda()
			
	def forward(self, x):
		# Expects data to be of shape (batch_size, num_frames*self.img_feature_dim)
		batch_size, _ = x.size()
		# Reshape to (batch_size, num_frames, img_feature_dim)
		x = x.view(batch_size, self.num_frames, self.img_feature_dim)
		# Reshape to (num_frames, batch_size, img_feature_dim), as expected by pytorch
		x = x.permute(1, 0, 2)
		# Discard intermediate hidden states
		_, last_hidden = self.lstm(x)
		# Output is of size (batch_size, hidden_size)
		return last_hidden[0]

