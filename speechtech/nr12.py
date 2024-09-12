import torch
from torchaudio.models import Conformer

#initialize conformer
conformer = Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=128,
    num_layers=4,
    depthwise_conv_kernel_size=31,
)

#random input
lengths = torch.randint(1,400,(10,))
input_dim = 80
input = torch.rand(
	10,
	int(lengths.max()),
	input_dim
)

#generate output
output = conformer(input,lengths)

#print
print(output[0].shape,output[1].shape)
