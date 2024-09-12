from TTS.api import TTS

TTS(
	"tts_models/en/ljspeech/tacotron2-DDC"
).tts_to_file(
	"This is me saying something.",
	file_path="intermediate.wav"
)
TTS(
	"voice_conversion_models/" +\
		"multilingual/vctk/freevc24"
).voice_conversion_to_file(
	source_wav="intermediate.wav",
	target_wav="/mhdata/quick.wav",
	file_path="mhoutput.wav"
)

