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

"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


		H_1 H_2 H_3 ... H_n
		  q   q   q	   q
			|  |   |	   |
			  \ |   |	  /
					  .....
				  \   |  /
						  a

Constructs a unit mapping.
	$$(H_1 + H_n, q) => (a)$$
	Where H is of `batch x n x dim` and q is of `batch x dim`.

	The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math

class GlobalAttention(nn.Module):
	def __init__(self, dim, dim_out):
		super(GlobalAttention, self).__init__()
		self.linear_in = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax()
		self.linear_out = nn.Linear(dim*2, dim_out, bias=False)
		self.tanh = nn.Tanh()
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, input, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

		# Get attention
		attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)
		attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

		weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
		contextCombined = torch.cat((weightedContext, input), 1)

		contextOutput = self.tanh(self.linear_out(contextCombined))

		return contextOutput, attn
