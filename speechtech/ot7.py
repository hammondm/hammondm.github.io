import torchaudio
from speechbrain.pretrained import \
	EncoderClassifier

classifier = EncoderClassifier.from_hparams(
	source="speechbrain/" + \
		"lang-id-commonlanguage_ecapa",
	savedir="/mhdata/ecapa",
	run_opts={"device":"cuda"}
)

_,_,_,lab = classifier.classify_file(
	'example-it.wav'
)

print(lab)

