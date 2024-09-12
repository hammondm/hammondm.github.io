#!/bin/bash

#smaller minilibrispeech setup: 30-90min
#(less if the data is already present)

nj=15

#copy things from mini_librispeech
cp -rL /opt/kaldi/egs/mini_librispeech/s5/utils/ .
cp -rL /opt/kaldi/egs/mini_librispeech/s5/local/ .
cp -rL /opt/kaldi/egs/mini_librispeech/s5/steps/ .
cp -rL /opt/kaldi/egs/mini_librispeech/s5/conf/ .

#directory for data
data=./corpus

#urls for data
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

#set general commands and path
. ./cmd.sh
. ./path.sh

#keep track of where we are
stage=0

#custom parsing of command-line options
. utils/parse_options.sh

#fail badly
set -euo pipefail

#make directory for data
mkdir -p $data

#get and unpack data
#dev-clean-2: 1089 files
#train-clean-5: 1519 files
echo Getting data
for part in dev-clean-2 train-clean-5; do
	local/download_and_untar.sh $data $data_url $part
done

#download language model: trigram model in 3 sizes
echo Downloading language model
if [ $stage -le 0 ]; then
	local/download_lm.sh $lm_url $data data/local/lm
fi

#format data; create aux files
if [ $stage -le 1 ]; then
	#replace hyphen with underline in dir names
	for part in dev-clean-2 train-clean-5; do
		local/data_prep.sh $data/LibriSpeech/$part \
			data/$(echo $part | sed s/-/_/g)
	done
	#make dictionary files
	local/prepare_dict.sh --stage 3 --nj ${nj} --cmd "$train_cmd" \
		data/local/lm data/local/lm data/local/dict_nosp
	#make language model files
	utils/prepare_lang.sh data/local/dict_nosp \
		"<UNK>" data/local/lang_tmp_nosp data/lang_nosp
	#make language model transducers
	local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 2 ]; then
	mfccdir=mfcc
	for part in dev_clean_2 train_clean_5; do
		#make MFCCs
		steps/make_mfcc.sh --cmd "$train_cmd" \
			--nj ${nj} data/$part exp/make_mfcc/$part $mfccdir
		#normalize MFCCs
		steps/compute_cmvn_stats.sh data/$part \
			exp/make_mfcc/$part $mfccdir
	done
	#get shortest 500 utterances
	utils/subset_data_dir.sh --shortest data/train_clean_5 \
		500 data/train_500short
fi

if [ $stage -le 3 ]; then
	#train monophone system on 500 sentences
	steps/train_mono.sh --boost-silence 1.25 --nj ${nj} \
		--cmd "$train_cmd" data/train_500short data/lang_nosp \
		exp/mono
	(
		#make lattice for small language model
		utils/mkgraph.sh data/lang_nosp_test_tgsmall \
			exp/mono exp/mono/graph_nosp_tgsmall
	)&
	#do something special for aligning silences
	steps/align_si.sh --boost-silence 1.25 --nj ${nj} \
		--cmd "$train_cmd" data/train_clean_5 data/lang_nosp \
		exp/mono exp/mono_ali_train_clean_5
fi

#train system on all utterances
if [ $stage -le 4 ]; then
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
		2000 10000 data/train_clean_5 data/lang_nosp \
		exp/mono_ali_train_clean_5 exp/tri1

	(
		#decode using tri1 model
		utils/mkgraph.sh data/lang_nosp_test_tgsmall \
			exp/tri1 exp/tri1/graph_nosp_tgsmall
		for test in dev_clean_2; do
			#acoustic model
			steps/decode.sh --nj ${nj} --cmd "$decode_cmd" \
				exp/tri1/graph_nosp_tgsmall \
				data/$test exp/tri1/decode_nosp_tgsmall_$test
			#two language models
			steps/lmrescore.sh --cmd "$decode_cmd" \
				data/lang_nosp_test_{tgsmall,tgmed} \
				data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
		done
	)&
fi

#wait until decoding is done
wait

