import os,torch,re
import torchaudio as ta

#sample input
wavfile = "quick.wav"

#move to GPU if possible
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print(device)

#load pretrained model
bundle = ta.pipelines.WAV2VEC2_ASR_BASE_960H
#move to GPU if possible
model = bundle.get_model().to(device)
print(model.__class__)

#load wave
waveform,sample_rate = ta.load(wavfile)
waveform = waveform.to(device)

#resample if needed
if sample_rate != bundle.sample_rate:
	waveform = ta.functional.resample(
		waveform,
		sample_rate,
		bundle.sample_rate
	)

#do inference
with torch.inference_mode():
	emission,_ = model(waveform)

#CTC decoding
class GreedyCTCDecoder(torch.nn.Module):
	def __init__(self,labels,ignore):
		super().__init__()
		self.labels = labels
		self.ignore = ignore
	def forward(self,emission: torch.Tensor):
		indices = torch.argmax(emission,dim=-1)
		indices = torch.unique_consecutive(
			indices,
			dim=-1
		)
		indices = [i for i in indices \
			if i not in self.ignore]
		return ''.join(
			[self.labels[i] for i in indices]
		)

#initialize decoder
decoder = GreedyCTCDecoder(
	labels=bundle.get_labels(),
	ignore=(0,1,2,3),
)
#decode
transcript = decoder(emission[0])
transcript = \
	re.sub('\\|',' ',transcript.lower())
#print result
print(transcript)
