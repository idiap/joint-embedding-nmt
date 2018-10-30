#    Copyright (c) OpenNMT-py, https://github.com/OpenNMT/OpenNMT-py
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

from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

	def __init__(self, srcData, tgtData, batchSize, cuda, volatile=False, sample_size=1, tgtVocab_size=0):
		self.src = srcData
		if tgtData:
			self.tgt = tgtData
			assert(len(self.src) == len(self.tgt))
		else:
			self.tgt = None
		self.cuda = cuda

		self.batchSize = batchSize
		self.numBatches = math.ceil(len(self.src)/batchSize)
		self.volatile = volatile
		self.sample_size = sample_size
		self.tgtSample = None
		self.indexSample = None
		self.tgtVocab_size = tgtVocab_size

	def _batchify(self, data, align_right=False, include_lengths=False):
		lengths = [x.size(0) for x in data]
		max_length = max(lengths)
		out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
		for i in range(len(data)):
			data_length = data[i].size(0)
			offset = max_length - data_length if align_right else 0
			out[i].narrow(0, offset, data_length).copy_(data[i])

		if include_lengths:
			return out, lengths
		else:
			return out

	def __getitem__(self, index):
		assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
		srcBatch, lengths = self._batchify(
			self.src[index*self.batchSize:(index+1)*self.batchSize],
			align_right=False, include_lengths=True)

		if self.tgt:
			tgt = self.tgtSample if self.tgtSample else self.tgt
			tgtBatch = self._batchify(
				tgt[index*self.batchSize:(index+1)*self.batchSize])
		else:
			tgtBatch = None

		# within batch sorting by decreasing length for variable length rnns
		indices = range(len(srcBatch))
		batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
		batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
		if tgtBatch is None:
			indices, srcBatch = zip(*batch)
		else:
			indices, srcBatch, tgtBatch = zip(*batch)

		def wrap(b, target=False):
			if b is None:
				return b
			b = torch.stack(b, 0).t().contiguous()
			if self.cuda:
				b = b.cuda()
			if target:
				b = Variable(b, volatile=self.volatile, requires_grad=False)
			else:
				b = Variable(b, volatile=self.volatile, requires_grad=False)
			return b

		# wrap lengths in a Variable to properly split it in DataParallel
		lengths = torch.LongTensor(lengths).view(1, -1)
		lengths = Variable(lengths, volatile=self.volatile)

		sampleBatch = None
		if self.indexSample:
			sampleBatch = self.indexSample[index]
			if self.cuda:
				sampleBatch = sampleBatch.cuda()
			sampleBatch = Variable(sampleBatch, volatile=self.volatile, requires_grad=False)

		return (wrap(srcBatch), lengths), wrap(tgtBatch, target=True), sampleBatch, indices

	def __len__(self):
		return self.numBatches


	def shuffle(self):
		data = list(zip(self.src, self.tgt))
		self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
		#if self.sample_size > 1:
			#self.sampling()
