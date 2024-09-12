import os

from trainer import Trainer,TrainerArgs

from TTS.tts.configs.shared_configs \
	import BaseDatasetConfig
from TTS.tts.configs.vits_config \
	import VitsConfig
from TTS.tts.datasets import \
	load_tts_samples
from TTS.tts.models.vits import \
	Vits,VitsAudioConfig
from TTS.tts.utils.text.tokenizer \
	import TTSTokenizer
from TTS.utils.audio import \
	AudioProcessor

output_path = os.path.dirname(
	os.path.abspath(__file__)
)
#location of data
dataset_config = BaseDatasetConfig(
	formatter="ljspeech",
	meta_file_train="metadata.csv",
	path=os.path.join(
		output_path,
		"/mhdata/LJSpeech-1.1/"
	)
)

#audio parameters
audio_config = VitsAudioConfig(
	sample_rate=22050,
	win_length=1024,
	hop_length=256,
	num_mels=80,
	mel_fmin=0,
	mel_fmax=None
)

#run parameters
config = VitsConfig(
	audio=audio_config,
	run_name="vits_ljspeech",
	#change for GPU limits
	batch_size=8,
	eval_batch_size=8,
	batch_group_size=5,
	num_loader_workers=2,
	num_eval_loader_workers=2,
	run_eval=True,
	test_delay_epochs=-1,
	epochs=2,
	#specific to English again
	text_cleaner="english_cleaners",
	use_phonemes=True,
	#specific to English again
	phoneme_language="en-us",
	phoneme_cache_path=os.path.join(
		output_path,
		"phoneme_cache"
	),
	compute_input_seq_cache=True,
	print_step=25,
	print_eval=True,
	mixed_precision=True,
	output_path=output_path,
	datasets=[dataset_config],
	cudnn_benchmark=False,
)

ap = AudioProcessor.init_from_config(
	config
)

tokenizer,config = \
	TTSTokenizer.init_from_config(config)

#get/partition data
train_samples,eval_samples = load_tts_samples(
	dataset_config,
	eval_split=True,
	eval_split_max_size= \
		config.eval_split_max_size,
	eval_split_size=config.eval_split_size,
)

#build model
model = Vits(
	config,ap,
	tokenizer,
	speaker_manager=None
)

#configure training
trainer = Trainer(
	TrainerArgs(),
	config,
	output_path,
	model=model,
	train_samples=train_samples,
	eval_samples=eval_samples,
)
#train
trainer.fit()
