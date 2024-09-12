import torch
from scipy.io.wavfile import write

#download pretrained tacotron2 model
tacotron2 = torch.hub.load(
	'NVIDIA/DeepLearningExamples:torchhub',
	'nvidia_tacotron2',
	model_math='fp16'
)
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()
#download pretrained waveglow vocoder
waveglow = torch.hub.load(
	'NVIDIA/DeepLearningExamples:torchhub',
	'nvidia_waveglow',
	model_math='fp16'
)
waveglow = waveglow.remove_weightnorm(
	waveglow
)
waveglow = waveglow.to('cuda')
waveglow.eval()
#prepare string for synthesis
text = "Hello there, how are you?"
utils = torch.hub.load(
	'NVIDIA/DeepLearningExamples:torchhub',
	'nvidia_tts_utils'
)
sequences,lengths = \
	utils.prepare_input_sequence([text])
#synthesize
with torch.no_grad():
	mel,_,_ = tacotron2.infer(
		sequences,
		lengths
	)
	audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050
#save
write("res.wav",rate,audio_numpy)

