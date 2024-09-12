import torch
import torch.nn as nn
import torch.nn.functional as F

#build a model
class RNNModel(nn.Module):
	def __init__(
		self,rnn_type,ntoken,ninp,
		nhid,nlayers,dropout=0.5
	):
		super(RNNModel,self).__init__()
		self.ntoken = ntoken
		#use dropout
		self.drop = nn.Dropout(dropout)
		#specify embedding layer
		self.encoder = nn.Embedding(ntoken,ninp)
		#specify RNN part
		self.rnn = getattr(nn,rnn_type)(
			ninp,nhid,nlayers,dropout=dropout
		)
		#decoder part
		self.decoder = nn.Linear(nhid,ntoken)
		#initialize weights
		self.init_weights()
		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers
	#function to initialize weights
	def init_weights(self):
		initrange = 0.1
		nn.init.uniform_(
			self.encoder.weight,
			-initrange,
			initrange
		)
		nn.init.zeros_(self.decoder.weight)
		nn.init.uniform_(
			self.decoder.weight,
			-initrange,
			initrange
		)
	#apply net
	def forward(self,input,hidden):
		#do dropbout
		emb = self.drop(self.encoder(input))
		#calculate output and hidden values
		output,hidden = self.rnn(emb,hidden)
		#do dropout for output too
		output = self.drop(output)
		#apply decoder
		decoded = self.decoder(output)
		#reorient
		decoded = decoded.view(-1,self.ntoken)
		return F.log_softmax(decoded,dim=1),hidden
	#function to initialize hidden values
	def init_hidden(self,bsz):
		weight = next(self.parameters())
		#extra values if LSTM
		if self.rnn_type == 'LSTM':
			return (
				weight.new_zeros(
					self.nlayers,
					bsz,
					self.nhid
				),
				weight.new_zeros(
					self.nlayers,
					bsz,
					self.nhid
				)
			)
		else:
			return weight.new_zeros(
				self.nlayers,
				bsz,
				self.nhid
			)
