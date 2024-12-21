
import torch
from datasets import load_dataset,DatasetDict
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers import WhisperForConditionalGeneration
from typing import Any,Dict,List,Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

#check for gpu
if torch.cuda.is_available():
    print('gpu')
else:
    print('no gpu')

#file location
loc = "cv-corpus-16.1-2023-12-06/cy/clips/"

#specify metadata files!
data_files = {'test':'tst.csv','train':'train.tsv'}

#load dataset
common_voice = DatasetDict()
common_voice["test"] = load_dataset(
    "csv",
    data_files=data_files,
    split="test",
    delimiter='\t',
    keep_default_na=False,
    data_dir=loc
)
common_voice["train"] = load_dataset(
    "csv",
    data_files=data_files,
    split="train",
    delimiter='\t',
    keep_default_na=False,
    data_dir=loc
)

#remove irrelevant fields
common_voice = common_voice.remove_columns([
    "client_id",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment"
])

#rename one column
common_voice = common_voice.rename_column('path','sound')

#get model
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small"
)

#get tokenizer
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Welsh",
    task="transcribe"
)

#massage audio
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Welsh",
    task="transcribe"
)

#specify where files are
def add_prefix(example):
    example['sound'] = loc + example['sound']
    return example
#actually add the location
common_voice['test'] = common_voice['test'].map(add_prefix)
common_voice['train'] = common_voice['train'].map(add_prefix)

#resample
common_voice['test'] = common_voice['test'].cast_column(
    "sound",
    Audio(sampling_rate=16000)
)
common_voice['train'] = common_voice['train'].cast_column(
    "sound",
    Audio(sampling_rate=16000)
)

def prepare_dataset(batch):
    audio = batch["sound"]
    #compute input features from audio
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    #encode text
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=20
)

#training and evaluation

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)

model.generation_config.language = "welsh"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

#Define a Data Collator

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(
            self,
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} \
                for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for \
                feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

#Evaluation Metrics

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#Define Training Configuration

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-cy",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    #eval_steps=1000,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

