import torch as t
import torch.nn as nn

epochs = 200
batchsize = 3
features = 13
classes = 5
inseqlen = 20
#input sequence length for CTC
T = 11
#longest target in batch (padding length)
S = 10
#minimum target length
S_min = 7

if t.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
print(f'Using {device}')

class W2L(nn.Module):
	def __init__(
			self,
			num_classes=40,
			input_type="waveform",
			num_features=1
	):
		super(W2L, self).__init__()
		if input_type == "waveform":
			acoustic_num_features = 250
		else:
			acoustic_num_features = num_features
		acoustic_model = nn.Sequential(
			nn.Conv1d(
				in_channels=acoustic_num_features,
				out_channels=250,
				kernel_size=48,
				stride=2,
				padding=23
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=250,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=250,
				out_channels=2000,
				kernel_size=32,
				stride=1,
				padding=16
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=2000,
				out_channels=2000,
				kernel_size=1,
				stride=1,
				padding=0
			),
			nn.ReLU(inplace=True),
			nn.Conv1d(
				in_channels=2000,
				out_channels=num_classes,
				kernel_size=1,
				stride=1,
				padding=0
			),
			nn.ReLU(inplace=True)
		)
		if input_type == "waveform":
			waveform_model = nn.Sequential(
				nn.Conv1d(
					in_channels=num_features,
					out_channels=250,
					kernel_size=250,
					stride=160,
					padding=45
				),
				nn.ReLU(inplace=True)
			)
			self.acoustic_model = nn.Sequential(
				waveform_model,
				acoustic_model
			)
		if input_type in ["power_spectrum",
			"mfcc"]:
			self.acoustic_model = acoustic_model
	def forward(self,x):
		x = self.acoustic_model(x)
		x = nn.functional.log_softmax(x,dim=1)
		return x

#instantiate model
w = W2L(
	num_classes=classes,
	input_type="mfcc",
	num_features=features
)
w.to(device)

#make optimizer
optimizer = \
	t.optim.Adam(w.parameters(),lr=0.0001)

#loss function
ctc_loss = nn.CTCLoss(
	zero_infinity=True
)

#random inputs, 20 frames long
inp = t.randn(batchsize,features,inseqlen)

inp = inp.to(device)

#initialize random targets
target = t.randint(
	low=1,
	high=classes,
	size=(batchsize,S),
	dtype=t.long
)

#length of each CTC input in batch
#(1x5 here)
input_lengths = t.full(
	size=(batchsize,),
	fill_value=T,
	dtype=t.long
)

target_lengths = t.randint(
	low=S_min,
	high=S,
	size=(batchsize,),
	dtype=t.long
)

target = target.to(device)
input_lengths = input_lengths.to(device)
target_lengths = target_lengths.to(device)

losses = []
for epoch in range(epochs):
	#zero grads
	optimizer.zero_grad()
	#calculate output
	outp = w(inp)
	#permute for loss
	outp = outp.permute(2,0,1)
	#calculate loss
	loss = ctc_loss(
		outp,
		target,
		input_lengths,
		target_lengths
	)
	print(f'epoch {epoch}, loss: {loss.item()}')
	losses.append(loss.item())
	#calculate gradients
	loss.backward()
	#stop exploding gradients!
	nn.utils.clip_grad_norm_(w.parameters(),1)
	#update weights
	optimizer.step()

print('\ntarget:\n',target.cpu().numpy(),'\n')

#decode outputs
print('decoding:')
for i in range(batchsize):
	print(
		outp[:,i,:].max(
			dim=1
		).indices.cpu().numpy()
	)

