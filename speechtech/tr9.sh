#!/usr/bin/env bash

#tweaked runfile for yes,no in Hebrew

train_cmd="utils/run.pl"
decode_cmd="utils/run.pl"

#get data
address=http://www.openslr.org/resources/1/
filename=waves_yesno.tar.gz
if [ ! -d waves_yesno ]; then
  wget ${address}${filename} || exit 1;
  tar -xvzf ${filename} || exit 1;
fi

#train and test names
train_yesno=train_yesno
test_base_name=test_yesno

#prepare data
local/prepare_data.sh waves_yesno
local/prepare_dict.sh
utils/prepare_lang.sh --position-dependent-phones false \
	data/local/dict "<SIL>" data/local/lang data/lang
local/prepare_lm.sh

#MFCCs and cmvn
for x in train_yesno test_yesno; do 
 steps/make_mfcc.sh --nj 1 data/$x exp/make_mfcc/$x mfcc
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
 utils/fix_data_dir.sh data/$x
done

#mono training
steps/train_mono.sh --nj 1 --cmd "$train_cmd" \
  --totgauss 400 \
  data/train_yesno data/lang exp/mono0a 
  
#make the WFST
utils/mkgraph.sh data/lang_test_tg exp/mono0a \
	exp/mono0a/graph_tgpr

#decode
steps/decode.sh --nj 1 --cmd "$decode_cmd" \
    exp/mono0a/graph_tgpr data/test_yesno \
	exp/mono0a/decode_test_yesno

#display word error rate
for x in exp/*/decode*; do
	[ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done

