import editdistance
from pyfoma import FST
import wfeatures as wf

filename = 'wikicym3.txt'

f = open(filename,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

rulelist = []
rulelist.append(FST.re("$^rewrite(g:ɡ)"))
rulelist.append(FST.re(
	"$^rewrite((si):ʃ / _ $sylplus)",
	{'sylplus':wf.sylplus}
))
rulelist.append(FST.re("$^rewrite((ái):(ai̯))"))
rulelist.append(FST.re("$^rewrite((ô):(oː))"))
rulelist.append(FST.re("$^rewrite((ŵ):(uː))"))
rulelist.append(FST.re("$^rewrite((nn):n)"))
rulelist.append(FST.re("$^rewrite((ï):(iː))"))
rulelist.append(FST.re("$^rewrite((ch):χ)"))
rulelist.append(FST.re("$^rewrite(c:k)"))
rulelist.append(FST.re("$^rewrite(n:ŋ / _ k)"))
rulelist.append(FST.re("$^rewrite(z:s)"))
rulelist.append(FST.re("$^rewrite((yw):(əu̯))"))
rulelist.append(FST.re(
	"$^rewrite(i:(i̯) / $sylplus _)",
	{'sylplus':wf.sylplus}
))
rulelist.append(FST.re("$^rewrite(t:d / s _)"))
rulelist.append(FST.re("$^rewrite(a:(aː) / # _ #)"))
rulelist.append(FST.re("$^rewrite((â):(a|(aː)) / _ #)"))
rulelist.append(FST.re("$^rewrite((â):(aː))"))
rulelist.append(FST.re("$^rewrite(y:ə)"))
rulelist.append(FST.re("$^rewrite((ff):F)"))
rulelist.append(FST.re("$^rewrite(f:v)"))
rulelist.append(FST.re("$^rewrite(F:f)"))
rulelist.append(FST.re(
	"$^rewrite(i:ɪ / _ $sylminus)",
	{'sylminus':wf.sylminus}
))
rulelist.append(FST.re(
	"$^rewrite(i:j / _ $sylplus)",
	{'sylplus':wf.sylplus}
))
rulelist.append(FST.re("$^rewrite(ê:(eː))"))
rulelist.append(FST.re(
	"$^rewrite(e:ɛ / _ $consplus)",
	{'consplus':wf.consplus}
))
rulelist.append(FST.re("$^rewrite(o:ɔ)"))
rulelist.append(FST.re("$^rewrite((aw):((au̯)|ɔ))"))
rulelist.append(FST.re(
	"$^rewrite(w:ʊ / _ $sylminus)",
	{'sylminus':wf.sylminus}
))
rulelist.append(FST.re("$^rewrite((th):θ)"))
rulelist.append(FST.re("$^rewrite((dd):ð)"))
rulelist.append(FST.re("$^rewrite((ph):f)"))
rulelist.append(FST.re("$^rewrite((ll):ɬ)"))
rulelist.append(FST.re("$^rewrite((rr):r)"))

rules = {}
for i in range(len(rulelist)):
	key = 'r' + str(i + 1)
	rules[key] = rulelist[i]

cs = ['$' + k for k in rules.keys()]
rulestring = ' @ '.join(cs)

everything = FST.re(rulestring,rules)

total = 0
for line in lines:
	word,trans = line.split('\t')
	word = word.lower()
	trans = ''.join(trans.split())
	outputs = list(everything.generate(word))
	d = 100
	for output in outputs:
		#print('\ttrying:',output)
		thisd = editdistance.eval(trans,output)
		if thisd < d:
			d = thisd
			bestoutput = output
	total += d
	print(f'{word} -> {bestoutput}, ({trans} - {d})')

print(f'total = {total} ({total/len(lines)})')

