#    Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#               Lesly Miculicich <lesly.miculicich@idiap.ch>
#
#    This file is part of joint-embedding-nmt which builds over OpenNMT-py
#    neural machine translation framework (https://github.com/OpenNMT/OpenNMT-py).
#
#    joint-embedding-nmt is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    joint-embedding-nmt is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with joint-embedding-nmt. If not, see http://www.gnu.org/licenses/

import torch, onmt
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
	def __init__(self, opt, dict_size, embedding, desc=False):
		super(Generator, self).__init__()
		self.full = True
		self.opt = opt
		self.desc = desc
		self.embedding = embedding

		# The parameters of the following linear unit are not used fully
		# by all models: e.g. Typical softmax and weight tying make use
		# of it but the joint models make use only its bias parameters
		self.linear = nn.Linear(opt.out_size, dict_size)

		if opt.generator in ['simple', 'tie']:
			if opt.generator == 'tie':
				 self.linear.weight = embedding.weight
		else:
 			if not hasattr(opt,"join_projection"):
				opt.join_projection = opt.word_vec_size_dec
			if not opt.onlyoutput and not opt.onlyinput and not opt.bilinear:
				self.full = True
				self.linear_emb = nn.Linear(opt.word_vec_size_dec, opt.join_projection)
				self.linear_hidden = nn.Linear(opt.out_size, opt.join_projection)
				if  opt.onlyoutput:
					self.linear_emb = nn.Linear(opt.word_vec_size_dec, opt.out_size)
					self.full = False
				if opt.onlyinput or opt.bilinear:
					self.linear_hidden = nn.Linear(opt.out_size, opt.word_vec_size_dec, bias=False)
					self.full = False
			else:
				self.linear_emb = nn.Linear(opt.word_vec_size_dec, opt.join_projection)
				self.linear_hidden = nn.Linear(opt.out_size, opt.join_projection)
			self.tanh = nn.Tanh()
			self.scaling = nn.Linear(1, opt.join_projection, bias=False)
			self.embedding = embedding
			self.dict_size = dict_size

		self.LogSoftmax = nn.LogSoftmax()

	def sample_join(self, sample):
		if sample is not None:
			weight = self.embedding.weight.index_select(0, sample)
			bias = self.linear.bias.index_select(0, sample)
			return weight, bias
		return self.embedding.weight, self.linear.bias.view(-1)

	def forward(self, hidden, sample=None):
		if self.opt.generator in ['simple', 'tie']:
			if sample is not None:
				weight = self.linear.weight.index_select(0, sample)
				bias = self.linear.bias.index_select(0, sample)
				v_out = torch.mm(hidden, weight.t()) + bias
			else:
				v_out = self.linear(hidden)
		else:
			embedding, bias = self.sample_join(sample)
			proj_hidden = hidden
			proj_emb = embedding
			if self.opt.onlyoutput or self.full:
				proj_emb = self.tanh(self.linear_emb(embedding))
			if self.opt.onlyinput or self.full:
				proj_hidden =  self.tanh(self.linear_hidden(hidden))
			if self.opt.bilinear:
				proj_hidden =  self.linear_hidden(hidden)
			v_out = torch.mm(proj_hidden, proj_emb.t()) + bias
		return self.LogSoftmax(v_out)
