import torch
from TTS.api import TTS

#get device
if torch.cuda.is_available():
	device = "cuda"
else:	
	device = "cpu"

#init TTS
tts = TTS(
	"tts_models/multilingual/" + \
		"multi-dataset/xtts_v2"
).to(device)

#run TTS
tts.tts_to_file(
	text="Hello world!",
	speaker_wav="quick.wav",
	language="en",
	file_path="mhoutput.wav"
)

