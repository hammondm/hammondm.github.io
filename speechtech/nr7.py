from transformers import pipeline

#test input
inp = \
	"This is a book about [MASK] technology."

#load pretrained model
unmasker = pipeline(
	'fill-mask',
	model='bert-base-uncased'
)

#calculate output
best = unmasker(inp)

#print results
print(f'\n{inp}')
for s in best:
	print(f"{s['score']:.3f}: {s['sequence']}")
