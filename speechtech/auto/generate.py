import torch,data

#location of corpus
par_data = '.'
#model checkpoint to use
par_checkpoint = \
	'/Users/hammond/Desktop/model.pt'
#output file for generated text
par_outf = \
	'/Users/hammond/Desktop/generated.txt'
#number of words to generate
par_words = 1000
#random seed
par_seed = 1111
#temperature >= 1e-3
par_temperature = 1.0
#reporting interval
par_log_interval = 100

#set random seed
torch.manual_seed(par_seed)

#switch to GPU if possible
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

with open(par_checkpoint,'rb') as f:
	model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(par_data)
ntokens = len(corpus.dictionary)

#initialize net
hidden = model.init_hidden(1)
input = torch.randint(
	ntokens,
	(1,1),
	dtype=torch.long
).to(device)

#generate output sequences
with open(par_outf,'w') as outf:
	with torch.no_grad():
		for i in range(par_words):
			output,hidden = model(input,hidden)
			word_weights = output.squeeze().div(
				par_temperature
			).exp().cpu()
			word_idx = torch.multinomial(
				word_weights,
				1
			)[0]
			input.fill_(word_idx)
			word = corpus.dictionary.idx2word[
				word_idx
			]
			outf.write(
				word + ('\n' if i % 20 == 19 else ' ')
			)
			if i % par_log_interval == 0:
				print(
					f'| Generated {i}/{par_words} words'
				)
