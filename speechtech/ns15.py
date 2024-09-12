import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#convert digits to tensors
transform = transforms.Compose(
	[transforms.ToTensor()]
)

#download MNIST dataset
path = '~/Desktop/datasets'
train_dataset = MNIST(
	path,
	transform=transform,
	download=True
)

#create dataloader
batch_size = 100
train_loader = DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True
)

#move to GPU if possible
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

#encoder model
class VAE(nn.Module):
	def __init__(
			self,
			input_dim=784,
			hidden_dim=400,
			latent_dim=200,
			device=device
	):
		super(VAE,self).__init__()
		#encoder: 2 layers
		self.encoder = nn.Sequential(
			nn.Linear(input_dim,hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim,latent_dim),
			nn.LeakyReLU(0.2)
			)
		#latent mean and variance
		self.mean_layer = nn.Linear(latent_dim,2)
		self.logvar_layer = \
			nn.Linear(latent_dim,2)
		#decoder: 3 layers
		self.decoder = nn.Sequential(
			nn.Linear(2,latent_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(latent_dim,hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim,input_dim),
			nn.Sigmoid()
			)
	#compute mean and variance in encoder
	def encode(self,x):
		x = self.encoder(x)
		mean = self.mean_layer(x)
		logvar = self.logvar_layer(x)
		return mean,logvar
	#do this so we can get the derivatives
	def reparameterization(self,mean,var):
		epsilon = torch.randn_like(var).to(device)
		z = mean + var*epsilon
		return z
	def decode(self,x):
		return self.decoder(x)
	def forward(self,x):
		mean,log_var = self.encode(x)
		z = self.reparameterization(
			mean,
			torch.exp(log_var)
		)
		x_hat = self.decode(z)
		return x_hat,mean,log_var

model = VAE().to(device)
optimizer = Adam(model.parameters(),lr=1e-3)

def loss_function(x,x_hat,mean,log_var):
	#output loss
	reproduction_loss = \
		nn.functional.binary_cross_entropy(
			x_hat,x,reduction='sum'
		)
	#KL divergence
	KLD = - 1 * torch.sum(
		1 + log_var - mean.pow(2) - log_var.exp()
	)
	#add them together
	return reproduction_loss + KLD

def train(
		model,
		optimizer,
		epochs,
		device,
		x_dim=784
):
	model.train()
	for epoch in range(epochs):
		overall_loss = 0
		for batch_idx,(x,_) in \
				enumerate(train_loader):
			x = x.view(batch_size,x_dim).to(device)
			optimizer.zero_grad()
			x_hat,mean,log_var = model(x)
			loss = loss_function(
				x,x_hat,mean,log_var
			)
			overall_loss += loss.item()
			loss.backward()
			optimizer.step()
		print(
			"\tEpoch:",
			epoch + 1,
			"\taverage loss: ",
			overall_loss/(batch_idx*batch_size)
		)
	return overall_loss

train(model,optimizer,epochs=20,device=device)

def generate(mean,var):
	z_sample = torch.tensor(
		[[mean,var]],
		dtype=torch.float
	).to(device)
	x_decoded = model.decode(z_sample)
	#reshape vector to 2d
	digit = \
		x_decoded.detach().cpu().reshape(28,28)
	plt.title(f'[{mean},{var}]')
	plt.imshow(digit,cmap='gray')
	plt.axis('off')
	plt.show()

generate(0.0,1.0)
