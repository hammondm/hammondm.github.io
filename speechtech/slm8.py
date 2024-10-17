from pyfoma import FST

#data
words = '''KAT cat
TAK tack
KOT coat
OK oak
OT oat
TO toe
AKT act
TAKT tact'''
words = words.split('\n')

#extract bits for WFSTs
letters = set()
pairs = []
for line in words:
    bits = line.split(' ')
    pairs.append((bits[0].lower(),bits[1]))
    for letter in bits[0]:
        letters.add(letter)
letters.add("''")

regexp = []
#arcs for acoustic model wfst
for let1 in letters:
	for let2 in letters:
		if let1 != "''" or let2 != "''":
			if let1 != let2:
				weight = '<1>'
			else:
				weight = ''
			#res = '((' + let1 + ':' + let2.lower() + ')' + weight + ')'
			res = '(' + let1 + ':' + let2.lower() + ')' + weight
			regexp.append(res)
regexp = '|'.join(regexp)
regexp = '(' + regexp + ')*'
am = FST.re(regexp)

print(f"am: {regexp}\n")

#language model arcs
arcs = []
for pair in pairs:
	lets = pair[0]
	token = pair[1]
	arc = lets[0] + ":'" + token + "'"
	for let in lets[1:]:
		arc += let + ":''"
	arc = '(' + arc + ')'
	arcs.append(arc)
arcs = '|'.join(arcs)
lm = FST.re(arcs)

print(f'lm: {arcs}\n')

#compose everything
fm = FST.re(
	"KATA @ $am @ $lm",
	{'am':am,'lm':lm}
)

#get best one
print(list(fm.words_nbest(1)))

