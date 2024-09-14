import os

from trainer import Trainer,TrainerArgs

from TTS.config.shared_configs import \
	BaseAudioConfig
from TTS.tts.configs.shared_configs \
	import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config \
	import Tacotron2Config
from TTS.tts.datasets import \
	load_tts_samples
from TTS.tts.models.tacotron2 \
	import Tacotron2
from TTS.utils.audio import \
	AudioProcessor
from TTS.tts.utils.text.tokenizer \
	import TTSTokenizer
from TTS.tts.configs.shared_configs \
	import CharactersConfig

output_path = os.path.dirname(
	os.path.abspath(__file__)
)
 
#welsh characters would be listed here!
characters = CharactersConfig(
	characters="list all characters here!",
	punctuations="list all punctuation here!",
	phonemes=None,
	is_unique=True,
)

#data location
dataset_config = BaseDatasetConfig(
	formatter="ljspeech",
	meta_file_train="onespeaker.csv",
	path=os.path.join(
		output_path,
		"/mhdata/mhcy/"
	)
)

#audio configuration
audio_config = BaseAudioConfig(
	sample_rate=16000,
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
config = Tacotron2Config(
	characters=characters,
	audio=audio_config,
	#adjust for GPU limits
	batch_size=8,
	eval_batch_size=8,
	num_loader_workers=4,
	num_eval_loader_workers=4,
	run_eval=True,
	test_delay_epochs=-1,
	ga_alpha=5.0,
	decoder_loss_alpha=0.25,
	postnet_loss_alpha=0.25,
	postnet_diff_spec_alpha=0,
	decoder_diff_spec_alpha=0,
	decoder_ssim_alpha=0,
	postnet_ssim_alpha=0,
	r=2,
	attention_type="dynamic_convolution",
	double_decoder_consistency=True,
	epochs=2,
	#Welsh is character-based since
	#no available g2p system
	text_cleaner="basic_cleaners",
	use_phonemes=False,
	print_step=25,
	print_eval=True,
	mixed_precision=False,
	output_path=output_path,
	datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)

tokenizer,config = \
	TTSTokenizer.init_from_config(
		config
)

#partition training data
train_samples,eval_samples = load_tts_samples(
	dataset_config,
	eval_split=True,
	eval_split_max_size=\
		config.eval_split_max_size,
	eval_split_size=config.eval_split_size,
)

#initialize model
model = Tacotron2(config,ap,tokenizer)

#initialize training
trainer = Trainer(
	TrainerArgs(),
	config,
	output_path,
	model=model,
	train_samples=train_samples,
	eval_samples=eval_samples,
	training_assets={"audio_processor": ap},
)
#train
trainer.fit()
