import os,glob,os,subprocess,tarfile
import wget,nemo,librosa,json
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import DictConfig

data_dir = '/data/an4'
config_path = 'quartzconf.yaml'
epochs = 30

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

#download data
if not os.path.exists(
		data_dir + '/an4_sphere.tar.gz'
	):
	an4_url = 'https://dldata-public.s3.us' + \
		'-east-2.amazonaws.com/an4_sphere.tar.gz'
	an4_path = wget.download(an4_url,data_dir)
else:
	an4_path = data_dir + '/an4_sphere.tar.gz'

#convert to wav files
if not os.path.exists(data_dir + '/an4/'):
	tar = tarfile.open(an4_path)
	tar.extractall(path=data_dir)
	sph_list = glob.glob(
		data_dir + '/an4/**/*.sph',
		recursive=True
	)
	for sph_path in sph_list:
		wav_path = sph_path[:-4] + '.wav'
		cmd = ["sox",sph_path,wav_path]
		subprocess.run(cmd)

#function to create manifest file
def build_manifest(
		transcripts_path,
		manifest_path,wav_path
	):
	with open(transcripts_path,'r') as fin:
		with open(manifest_path,'w') as fout:
			for line in fin:
				transcript = line[: \
					line.find('(')-1].lower()
				transcript = transcript.replace(
					'<s>',''
				).replace('</s>','')
				transcript = transcript.strip()
				file_id = line[line.find('(')+1 : -2]
				audio_path = os.path.join(
					data_dir,wav_path,
					file_id[file_id.find('-')+1 : \
						file_id.rfind('-')],
					file_id + '.wav')
				duration = librosa.core.get_duration(
					filename=audio_path
				)
				metadata = {
					"audio_filepath": audio_path,
					"duration": duration,
					"text": transcript
				}
				json.dump(metadata,fout)
				fout.write('\n')
				
#make manifest files
train_transcripts = data_dir + \
	'/an4/etc/an4_train.transcription'
train_manifest = data_dir + \
	'/an4/train_manifest.json'
if not os.path.isfile(train_manifest):
	build_manifest(
		train_transcripts,
		train_manifest,
		'an4/wav/an4_clstk'
	)
test_transcripts = data_dir + \
	'/an4/etc/an4_test.transcription'
test_manifest = data_dir + \
	'/an4/test_manifest.json'
if not os.path.isfile(test_manifest):
	build_manifest(
		test_transcripts,
		test_manifest,
		'an4/wav/an4test_clstk'
	)

#read config from yaml file
yaml = YAML(typ='safe')
with open(config_path) as f:
	params = yaml.load(f)

print(params)

#build trainer
trainer = pl.Trainer(
	devices=1,
	accelerator='gpu',
	max_epochs=epochs
)

#specify training and validation data
params['model']['train_ds']\
	['manifest_filepath'] = train_manifest
params['model']['validation_ds']\
	['manifest_filepath'] = test_manifest

#build model
first_asr_model = \
	nemo_asr.models.EncDecCTCModel(
		cfg=DictConfig(params['model']),
		trainer=trainer
)

#train
trainer.fit(first_asr_model)

#do some inference
paths2audio_files = [
		os.path.join(
			data_dir,
			'an4/wav/an4_clstk/mgah/cen2-mgah-b.wav'
		),
		os.path.join(
			data_dir,
			'an4/wav/an4_clstk/fmjd/cen7-fmjd-b.wav'
		),
		os.path.join(
			data_dir,
			'an4/wav/an4_clstk/fmjd/cen8-fmjd-b.wav'
		),
		os.path.join(
			data_dir,
			'an4/wav/an4_clstk/fkai/cen8-fkai-b.wav'
		)
	]
print(first_asr_model.transcribe(
	paths2audio_files=paths2audio_files,
	batch_size=4
))

