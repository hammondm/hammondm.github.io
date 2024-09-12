import coqui_stt_training
from coqui_stt_training.util.config \
	import initialize_globals_from_args
from coqui_stt_training.train import train
from coqui_stt_training.evaluate import test

pfx = \
'/mhdata/cv-corpus-16.0-2023-12-06/cy/clips/'

initialize_globals_from_args(
	#file locations
	train_files=[pfx+"train.csv"],
	dev_files=[pfx+"dev.csv"],
	test_files=[pfx+"tst.csv"],
	checkpoint_dir="welsh/checkpoints/",
	load_train="init",
	#size
	n_hidden=200,
	epochs=3,
	beam_width=1,
	#adjust these fof GPU/CPU limits
	train_batch_size=128,
	dev_batch_size=128,
	test_batch_size=10,
	skip_batch_test=True
)

train()
test()
