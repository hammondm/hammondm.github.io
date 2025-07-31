import time,os,torch
import torch.nn as nn
import data,model

#location of data corpus
par_data = '.'
#path to save final model
par_save = '/Users/hammond/Desktop/model.pt'

#epochs to train
par_epochs = 7
#batch size for validation and test
eval_batch_size = 10
#size of word embeddings
par_emsize = 200
#number of hidden units per layer
par_nhid = 200
#number of layers
par_nlayers = 2
#initial learning rate
par_lr = 20
#gradient clipping
par_clip = 0.25
#batch size
par_batch_size = 20
#sequence length
par_bptt = 35
#dropout for layers (0 = no dropout)
par_dropout = 0.2
#random seed
par_seed = 1111
#report interval
par_log_interval = 200
#set random seed
torch.manual_seed(par_seed)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

#get data
corpus = data.Corpus(par_data)

#break data into batches
def tobatches(data,bsz):
	nbatch = data.size(0) // bsz
	data = data.narrow(0,0,nbatch * bsz)
	data = data.view(bsz,-1).t().contiguous()
	return data.to(device)

#get train, validation and test data
train_data = tobatches(
	corpus.train,
	par_batch_size
)
val_data = tobatches(
	corpus.valid,
	eval_batch_size
)
test_data = tobatches(
	corpus.test,
	eval_batch_size
)

ntokens = len(corpus.dictionary)

#make model
model = model.RNNModel(
	'LSTM',
	ntokens,
	par_emsize,
	par_nhid,
	par_nlayers,
	par_dropout
).to(device)

#negative log likelihood loss
criterion = nn.NLLLoss()

#detach hidden states from their history
def repackage_hidden(h):
	if isinstance(h,torch.Tensor):
		return h.detach()
	else:
		return tuple(
			repackage_hidden(v) for v in h
		)

#split each item into par_bptt lengths
def get_batch(source,i):
	seq_len = min(par_bptt,len(source) - 1 - i)
	data = source[i:i+seq_len]
	target = source[i+1:i+1+seq_len].view(-1)
	return data,target

#calculate loss for a set of items
def evaluate(data_source):
	model.eval()
	total_loss = 0.
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(eval_batch_size)
	with torch.no_grad():
		for i in range(
			0,
			data_source.size(0)-1,
			par_bptt
		):
			data,targets = get_batch(data_source,i)
			output,hidden = model(data,hidden)
			hidden = repackage_hidden(hidden)
			total_loss += len(data) * \
				criterion(output,targets).item()
	return total_loss / (len(data_source) - 1)

#training function
def train():
	model.train()
	total_loss = 0.
	start_time = time.time()
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(par_batch_size)
	for batch,i in enumerate(range(
		0,
		train_data.size(0)-1,
		par_bptt
	)):
		data,targets = get_batch(train_data,i)
		model.zero_grad()
		hidden = repackage_hidden(hidden)
		output,hidden = model(data,hidden)
		#calculate loss
		loss = criterion(output,targets)
		loss.backward()
		#clip gradients
		torch.nn.utils.clip_grad_norm_(
			model.parameters(),
			par_clip
		)
		#update weights
		for p in model.parameters():
			p.data.add_(p.grad,alpha=-lr)
		total_loss += loss.item()
		#display/log current values
		if batch % par_log_interval == 0 \
			and batch > 0:
			cur_loss = total_loss / par_log_interval
			elapsed = time.time() - start_time
			bdenom = len(train_data) // par_bptt
			msb = elapsed * 1000 / par_log_interval
			print(
				f'| epoch {epoch:3d} | ' +
				f'{batch:5d}/{bdenom:5d} ' +
				f'batches | lr {lr:02.2f}' + 
				' | ms/batch ' +
				f'{msb:5.2f} | loss {cur_loss:5.2f}'
			)
			total_loss = 0
			start_time = time.time()

#Loop over epochs
lr = par_lr
best_val_loss = None

try:
	for epoch in range(1,par_epochs+1):
		epoch_start_time = time.time()
		train()
		val_loss = evaluate(val_data)
		tm = (time.time() - epoch_start_time)
		#end of epoch display values
		print('-' * 79)
		print(
			f'| end of epoch {epoch:3d} | ' +
			f'time: {tm:5.2f}s | ' +
			f'valid loss {val_loss:5.2f}'
		) 
		print('-' * 79)
		#save model if validation loss is best
		if not best_val_loss or \
			val_loss < best_val_loss:
			with open(par_save,'wb') as f:
				torch.save(model,f)
			best_val_loss = val_loss
		else:
			#anneal lr if no improvement
			#in validation
			lr /= 4.0
except KeyboardInterrupt:
	print('-' * 79)
	print('Exiting from training early')

#load best saved model
with open(par_save,'rb') as f:
	model = torch.load(f)
	model.rnn.flatten_parameters()

#run on test data
test_loss = evaluate(test_data)
print('=' * 79)
print(
	f'| End of training | test loss ' +
	f'{test_loss:5.2f}'
)
print('=' * 79)

