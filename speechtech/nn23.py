import sys
sys.dont_write_bytecode = True

import time,math,os,torch
import torch.nn as nn
import torch.nn.functional as F
from io import open

#location of corpus
argdata = '/Users/hammond/Desktop/data/wikitext-2'
#size of word embeddings
argemsize = 200
#hidden units per layer
argnhid = 200
#number of layers
argnlayers = 2
#initial learning rate
#arglr = 20
arglr = 5
#gradient clipping
argclip = 0.25
#max epochs
argepochs = 1
#batch size
argbatch_size = 20
#sequence length
argbptt = 35
#dropout (0 = no dropout)
argdropout = 0.2
#random seed
argseed = 666
#use CUDA
argcuda = False
#mac GPU
argmps = True
#reporting interval
arglog_interval = 200
#where to save model'
argsave = '/Users/hammond/Desktop/model.pt'
#number of attention heads
argnhead = 2

#separate model for sinusoidal encoding
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * \
            (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)

#tweak basic transformer class
class TransformerModel(nn.Transformer):
    def __init__(self,ntoken,ninp,nhead,nhid,nlayers,dropout=0.5):
        super(TransformerModel,self).__init__(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            num_encoder_layers=nlayers
        )
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp,dropout)
        self.input_emb = nn.Embedding(ntoken,ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp,ntoken)
        self.init_weights()
    def _generate_square_subsequent_mask(self,sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight,-initrange,initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight,-initrange,initrange)
    def forward(self,src,has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(
                    len(src)
                ).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src,mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output,dim=-1)

#set random seed
torch.manual_seed(argseed)

use_mps = argmps and torch.backends.mps.is_available()
if argcuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self,word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self,path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path,'train.txt'))
        self.valid = self.tokenize(os.path.join(path,'valid.txt'))
        self.test = self.tokenize(os.path.join(path,'test.txt'))
    def tokenize(self,path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        #add words to dictionary
        with open(path,'r',encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        #tokenize
        with open(path,'r',encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids

corpus = Corpus(argdata)

def batchify(data, bsz):
    #divide data into bsz parts
    nbatch = data.size(0) // bsz
    #trim off remainders
    data = data.narrow(0, 0, nbatch * bsz)
    #divide data across bsz batches
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, argbatch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

model = TransformerModel(
	ntokens,
	argemsize,
	argnhead,
	argnhid,
	argnlayers,
	argdropout
).to(device)

criterion = nn.NLLLoss()

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(argbptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, argbptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, argbptt)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), argclip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        total_loss += loss.item()
        if batch % arglog_interval == 0 and batch > 0:
            cur_loss = total_loss / arglog_interval
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} |'
                ' ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // argbptt, lr,
                elapsed * 1000 / arglog_interval,cur_loss,math.exp(cur_loss))
            )
            total_loss = 0
            start_time = time.time()

#loop over epochs
lr = arglr
best_val_loss = None

try:
    for epoch in range(1, argepochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch,(time.time()-epoch_start_time),
                                           val_loss,math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(argsave, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            #anneal learning rate if no improvement in validation data
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

#load best model
with open(argsave, 'rb') as f:
    safe_globals = [
        PositionalEncoding,
        TransformerModel,
        torch.nn.functional.relu,
        torch.nn.modules.activation.MultiheadAttention,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.dropout.Dropout,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
        torch.nn.modules.normalization.LayerNorm,
        torch.nn.modules.sparse.Embedding,
        torch.nn.modules.transformer.TransformerEncoder,
        torch.nn.modules.transformer.TransformerEncoderLayer,
    ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f,weights_only=False)

test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

