import os

from trainer import Trainer,TrainerArgs

from TTS.config import BaseAudioConfig, \
	BaseDatasetConfig
from TTS.tts.configs.fast_speech_config \
	import FastSpeechConfig
from TTS.tts.datasets import \
	load_tts_samples
from TTS.tts.models.forward_tts \
	import ForwardTTS
from TTS.tts.utils.text.tokenizer \
	import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

output_path = os.path.dirname(
	os.path.abspath(__file__)
)

#data location
dataset_config = BaseDatasetConfig(
	formatter="ljspeech",
	meta_file_train="metadata.csv",
	path=os.path.join(
		output_path,
		"/mhdata/LJSpeech-1.1/"
	),
)

#audio features
audio_config = BaseAudioConfig(
	sample_rate=22050,
	do_trim_silence=True,
	trim_db=60.0,
	signal_norm=False,
	mel_fmin=0.0,
	mel_fmax=8000,
	spec_gain=1.0,
	log_func="np.log",
	ref_level_db=20,
	preemphasis=0.0,
)

#training parameters
config = FastSpeechConfig(
	run_name="fast_speech_ljspeech",
	audio=audio_config,
	#adjust these for GPU limits
	batch_size=16,
	eval_batch_size=16,
	num_loader_workers=4,
	num_eval_loader_workers=2,
	compute_input_seq_cache=True,
	compute_f0=False,
	run_eval=True,
	test_delay_epochs=-1,
	epochs=3,
	#specific to English
	text_cleaner="english_cleaners",
	use_phonemes=True,
	#specific to English
	phoneme_language="en-us",
	phoneme_cache_path=os.path.join(
		output_path,
		"phoneme_cache"
	),
	precompute_num_workers=8,
	print_step=50,
	print_eval=False,
	mixed_precision=False,
	max_seq_len=500000,
	output_path=output_path,
	datasets=[dataset_config],
)

if not config.model_args.use_aligner:
	manager = ModelManager()
	model_path,config_path,_ = \
		manager.download_model(
		"tts_models/en/ljspeech/tacotron2-DCA"
	)
	#stand-alone code for attention masks
	attcommand = f'''python \\
		TTS/bin/compute_attention_masks.py \\
		--model_path {model_path} \\
		--config_path {config_path} \\
		--dataset ljspeech \\
		--dataset_metafile metadata.csv \\
		--data_path \\
		./recipes/ljspeech/LJSpeech-1.1/ \\
		--use_cuda true'''

	os.system(attcommand)

ap = AudioProcessor.init_from_config(config)

tokenizer,config = \
	TTSTokenizer.init_from_config(config)

train_samples,eval_samples = load_tts_samples(
	dataset_config,
	eval_split=True
)

model = ForwardTTS(config,ap,tokenizer)

#initialize training
trainer = Trainer(
	TrainerArgs(),
	config,
	output_path,
	model=model,
	train_samples=train_samples,
	eval_samples=eval_samples
)
#train
trainer.fit()
