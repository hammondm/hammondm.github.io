
#added first two lines below
import os

#specify database location
os.environ["PYANNOTE_DATABASE_CONFIG"] = \
'/AMI-diarization-setup/pyannote/database.yml'

from pyannote.database import \
	get_protocol, FileFinder
from pyannote.audio.tasks import \
	VoiceActivityDetection
from pyannote.audio.models.segmentation \
	import PyanNet
import pytorch_lightning as pl
from pyannote.audio import Inference
from pyannote.audio.pipelines \
	import VoiceActivityDetection as \
	VoiceActivityDetectionPipeline
from pyannote.metrics.detection import \
	DetectionErrorRate

preprocessors = {"audio": FileFinder()}
ami = get_protocol(
	'AMI.SpeakerDiarization.only_words',
	preprocessors=preprocessors
)

#set basic parameters
vad = VoiceActivityDetection(
	ami,
	duration=2.,
	batch_size=8
)

#define NN
model = PyanNet(
	sincnet={'stride': 10},
	task=vad
).to('cuda')

#set up training
trainer = pl.Trainer(
	devices=1,
	accelerator="gpu",
	max_epochs=1
)
#train
trainer.fit(model)

#get test data
test_file = next(ami.test())

#test
inference = Inference(model)
vad_probability = inference(test_file)

expected_output = \
	test_file["annotation"].\
	get_timeline().support()
print('expected output:')
print(expected_output)

#set thresholds manually for prediction spans
pipeline = VoiceActivityDetectionPipeline(
	segmentation=model
)
initial_params = {
	"onset": 0.6,
	"offset": 0.4, 
	"min_duration_on": 0.0,
	"min_duration_off": 0.0
}

pipeline.instantiate(initial_params)

#predicted values
print('\npredicted spans:')
print(pipeline(test_file).get_timeline())

metric = DetectionErrorRate()

for file in ami.test():
	speech = pipeline(file)
	_ = metric(
		file['annotation'],
		speech,
		uem=file['annotated']
	)
    
#aggregate performance over whole test set
der = abs(metric)
print(f'error rate = {der * 100:.1f}%')

