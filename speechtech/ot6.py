import os,wget,librosa,json
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from nemo.collections.asr.parts.\
	utils.speaker_utils import \
	rttm_to_labels, \
	labels_to_pyannote_object
from nemo.collections.asr.models \
	import ClusteringDiarizer

#location of data and results
ROOT = '/mhdata/'
data_dir = os.path.join(ROOT,'an4data')
os.makedirs(data_dir,exist_ok=True)
an4_audio = os.path.join(
	data_dir,
	'an4_diarize_test.wav'
)
an4_rttm = os.path.join(
	data_dir,
	'an4_diarize_test.rttm'
)

#download data
if not os.path.exists(an4_audio):
	an4_audio_url = \
		"https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
	an4_audio = wget.download(
		an4_audio_url,
		data_dir
	)
if not os.path.exists(an4_rttm):
	an4_rttm_url = \
		"https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
	an4_rttm = wget.download(
		an4_rttm_url,
		data_dir
	)

#load sound file
sr = 16000
signal,sr = librosa.load(an4_audio,sr=sr)
#load labels
labels = rttm_to_labels(an4_rttm)
reference = labels_to_pyannote_object(labels)

#create metadata file
meta = {
	'audio_filepath': an4_audio,
	'offset': 0,
	'duration':None,
	'label': 'infer',
	'text': '-',
	'num_speakers': 2,
	'rttm_filepath': an4_rttm,
	'uem_filepath' : None
}
fname = data_dir+'/input_manifest.json'
with open(fname,'w') as fp:
	json.dump(meta,fp)
	fp.write('\n')

#VAD subdirectory
output_dir = os.path.join(
	data_dir,
	'oracle_vad'
)
os.makedirs(output_dir,exist_ok=True)

#yaml file for experiment
MODEL_CONFIG = os.path.join(
	data_dir,
	'diar_infer_telephonic.yaml'
)
if not os.path.exists(MODEL_CONFIG):
	config_url = \
		"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
	MODEL_CONFIG = wget.download(
		config_url,
		data_dir
	)
config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

#load pretrained TitaNet
config.diarizer.manifest_filepath = \
	data_dir + '/input_manifest.json'
config.diarizer.out_dir = output_dir
pretrained_speaker_model = 'titanet_large'
config.diarizer.speaker_embeddings.\
	model_path = pretrained_speaker_model
config.diarizer.speaker_embeddings.\
	parameters.window_length_in_sec = \
	[1.5,1.25,1.0,0.75,0.5]
config.diarizer.speaker_embeddings.\
	parameters.shift_length_in_sec = \
	[0.75,0.625,0.5,0.375,0.1]
config.diarizer.speaker_embeddings.\
	parameters.multiscale_weights= \
	[1,1,1,1,1]
config.diarizer.oracle_vad = True
config.diarizer.clustering.parameters.\
	oracle_num_speakers = False

#use given VAD
oracle_vad_clusdiar_model = \
	ClusteringDiarizer(cfg=config)
oracle_vad_clusdiar_model.diarize()

print(labels)
print(reference)

fname = data_dir + \
	'/oracle_vad/pred_rttms/an4_diarize_test.rttm'
f = open(fname,'r')
t = f.read()
f.close()
print(t)

