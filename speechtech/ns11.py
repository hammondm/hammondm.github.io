from sklearn import datasets
from sklearn.preprocessing import \
	StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
from torch.distributions.\
	multivariate_normal \
	import MultivariateNormal

epochs = 30

#make data
n_samples = 2000
#make moon shapes
moons = datasets.make_moons(
	n_samples=n_samples,
	noise=.05
)
X,y = moons
#normalize moons
X = StandardScaler().fit_transform(X)

#plot data:
plt.scatter(X[:,0],X[:,1])
plt.title('Not a normal distribution')
plt.show()

#make multivariate normal distribution
base_mu = torch.zeros(2)
base_cov = torch.eye(2)
base_dist = MultivariateNormal(
	base_mu,
	base_cov
)
Z = base_dist.rsample(
	sample_shape=(3000,)
)

#plot data
plt.scatter(Z[:,0],Z[:,1])
plt.title('A normal distribution')
plt.show()

#neural net to try different functions:
#need values for sigma and mu
class R_NVP(nn.Module):
	def __init__(self,hidden):
		super().__init__()
		self.sig_net = nn.Sequential(
			nn.Linear(1,hidden),
			nn.LeakyReLU(),
			nn.Linear(hidden,1)
		)
		self.mu_net = nn.Sequential(
			nn.Linear(1,hidden),
			nn.LeakyReLU(),
			nn.Linear(hidden,1)
		)
	def forward(self,x):
		x1,x2 = x[:,:1],x[:,1:] 
		sig = self.sig_net(x1)
		#r-nvp calculation
		z1,z2 = x1, x2 * \
			torch.exp(sig) + self.mu_net(x1)
		z_hat = torch.cat([z1,z2],dim=-1)
		#need these to calculate loss
		log_pz = base_dist.log_prob(z_hat)
		log_jacob = sig.sum(-1)
		return z_hat,log_pz,log_jacob
	def inverse(self,Z):
		#so we can go in the other direction
		z1,z2 = Z[:,:1], Z[:,1:] 
		x1 = z1
		x2 = (z2 - self.mu_net(z1)) * \
			torch.exp(-self.sig_net(z1))
		return torch.cat([x1,x2],-1)

#make model
model = R_NVP(hidden=512)
optim = torch.optim.Adam(
	model.parameters(),
	lr=1e-3
)
scheduler = torch.optim.lr_scheduler.\
	ExponentialLR(optim,0.999)
n_samples = 512

#train
batch_size = n_samples
losses = []
for _ in range(epochs):
	#get some sample moons
	X,_ = datasets.make_moons(
		n_samples=batch_size,
		noise=.05
	)
	#scale them
	X = torch.from_numpy(
		StandardScaler().fit_transform(X)
	).float()
	optim.zero_grad()
	#calculate loss
	z,log_pz,log_jacob = model(X)
	loss = (-log_pz - log_jacob).mean()
	losses.append(loss.detach())
	#update
	loss.backward()
	optim.step()
	scheduler.step()

#visualize
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
X_hat = model.inverse(Z).detach().numpy()
plt.scatter(X_hat[:,0],X_hat[:,1])
plt.title(
	"Inverse of Normal Samples Z: X = F^-1(Z)"
)
plt.show()
n_samples = 3000
X,_ = datasets.make_moons(
	n_samples=n_samples,
	noise=.05
)
X = torch.from_numpy(
	StandardScaler().fit_transform(X)
).float()
z,_,_ = model(X)
z = z.detach().numpy()
plt.scatter(z[:,0],z[:,1])
plt.title(
	"Transformation of Samples X: Z = F(X)"
)
plt.show()

