import torch
from transformers import pipeline
from evaluate import load

pfx = 'cv-corpus-16.1-2023-12-06/cy/clips/'

wermetric = load('wer')

whisper = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-small",
    torch_dtype=torch.float16,
    device="cuda:0"
)

f = open(pfx+'tst.csv','r')
t = f.read()
f.close()

t = t.split('\n')
t = t[1:-1]

refs = []
preds = []

for line in t:
    fields = line.split('\t')
    print(f'{fields[1]}: {fields[2]}')
    refs.append(fields[2])
    transcription = whisper(
        pfx+fields[1],
        generate_kwargs={"language": "welsh"}
    )
    preds.append(transcription['text'])

wer = wermetric.compute(
    references=refs,
    predictions=preds
)
print(wer*100)

