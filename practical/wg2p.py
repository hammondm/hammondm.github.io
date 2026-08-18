import editdistance,random
from pyfoma import FST
import wfeatures as wf

filename = 'wikicym3.txt'

f = open(filename,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

rulelist = []

#dummy initial rule, comment out once you have real rules
#rulelist.append(FST.re("$^rewrite(a:a)"))

#total = 23206 (3.394178733362586)
rulelist.append(FST.re("$^rewrite(j:(d͡ʒ))"))
#total = 23128 (3.382770220857101)
rulelist.append(FST.re("$^rewrite((ng):ŋ)"))
#total = 22619 (3.3083223636097703)
rulelist.append(FST.re("$^rewrite(g:ɡ)"))
#total = 21031 (3.0760567500365656)
rulelist.append(FST.re("$^rewrite(u:ɨ)"))
#total = 21018 (3.0741553312856515)
rulelist.append(FST.re(
	"$^rewrite(w:u / $sylminus _ #)",
	{'sylminus':wf.sylminus}
))
#total = 21004 (3.0721076495538977)
rulelist.append(FST.re("$^rewrite((iw):(ɪu̯) / _ #)"))
#total = 20930 (3.0612841889717712)
rulelist.append(FST.re("$^rewrite((ae):(ei̯))"))
#total = 20899 (3.056750036565745)
rulelist.append(FST.re("$^rewrite((sh):ʃ)"))
#total = 20871 (3.052654673102238)
rulelist.append(FST.re("$^rewrite((ò):ɔ)"))
#total = 20866 (3.05192335819804)
rulelist.append(FST.re("$^rewrite((nh):(n̥))"))
#total = 20856 (3.0504607283896448)
rulelist.append(FST.re("$^rewrite((mh):(m̥))"))
#total = 20801 (3.0424162644434696)
rulelist.append(FST.re("$^rewrite((ŋh):(ŋ̊))"))
#total = 20747 (3.0345180634781337)
rulelist.append(FST.re("$^rewrite((ew):(ɛu̯))"))
#total = 20508 (2.999561211057481)
rulelist.append(FST.re("$^rewrite((iw):(jʊ) / _ [mr] #)"))
#total = 20272 (2.9650431475793475)
rulelist.append(FST.re(
	"$^rewrite((si):ʃ / _ $sylplus)",
	{'sylplus':wf.sylplus}
))
#total = 20167 (2.949685534591195)
rulelist.append(FST.re("$^rewrite((ái):(ai̯))"))
#total = 20161 (2.948807956706158)
rulelist.append(FST.re("$^rewrite((ô):(oː))"))
#total = 20064 (2.9346204475647215)
rulelist.append(FST.re("$^rewrite((ŵ):(uː))"))
#total = 19976 (2.921749305250841)
rulelist.append(FST.re("$^rewrite((nn):n)"))
#total = 19806 (2.8968845985081177)
rulelist.append(FST.re("$^rewrite((ï):(iː))"))
#total = 19741 (2.887377504753547)
rulelist.append(FST.re("$^rewrite((ch):χ)"))
#total = 18546 (2.7125932426502852)
rulelist.append(FST.re("$^rewrite(c:k)"))
#total = 17202 (2.5160157964019305)
rulelist.append(FST.re("$^rewrite(n:ŋ / _ k)"))
#total = 17127 (2.5050460728389643)
rulelist.append(FST.re("$^rewrite(z:s)"))
#total = 17126 (2.504899809858125) *
rulelist.append(FST.re("$^rewrite((yw):(əu̯))"))
#total = 17060 (2.4952464531227148)
rulelist.append(FST.re(
	"$^rewrite(i:(i̯) / $sylplus _)",
	{'sylplus':wf.sylplus}
))
#total = 16470 (2.4089512944273803)
rulelist.append(FST.re("$^rewrite(t:d / s _)"))
#total = 16320 (2.387011847301448)
rulelist.append(FST.re("$^rewrite((â):(aː))"))
#total = 16174 (2.3656574520988736)
rulelist.append(FST.re("$^rewrite((ff):F)"))
#total = 16194 (2.3685827117156646) *
rulelist.append(FST.re("$^rewrite(f:v)"))
#total = 15243 (2.2294866169372534)
rulelist.append(FST.re("$^rewrite(F:f)"))
#total = 14896 (2.1787333625859295)
rulelist.append(FST.re(
	"$^rewrite(i:ɪ / _ $sylminus)",
	{'sylminus':wf.sylminus}
))
#total = 13999 (2.0475354687728538)
rulelist.append(FST.re(
	"$^rewrite(i:j / _ $sylplus)",
	{'sylplus':wf.sylplus}
))
#total = 13075 (1.9123884744771098)
rulelist.append(FST.re("$^rewrite(ê:(eː))"))
#total = 12962 (1.8958607576422408)
rulelist.append(FST.re(
	"$^rewrite(e:ɛ / _ $consplus)",
	{'consplus':wf.consplus}
))
#total = 11148 (1.630539710399298)
rulelist.append(FST.re("$^rewrite(o:ɔ)"))
#total = 8657 (1.2661986251279802)
rulelist.append(FST.re("$^rewrite((aw):(au̯))"))
#total = 8174 (1.1955536053824778)
rulelist.append(FST.re(
	"$^rewrite(w:ʊ / _ $sylminus)",
	{'sylminus':wf.sylminus}
))
#total = 7556 (1.105163083223636)
rulelist.append(FST.re("$^rewrite((ỳ):ə)"))
#total = 7552 (1.104578031300278)
rulelist.append(FST.re(
	"$^rewrite(y:ɨ / _ $c* #)",
	{'c':wf.sylminus}
))
#total = 7537 (1.1023840865876846)
rulelist.append(FST.re("$^rewrite(y:ə)"))
#total = 6302 (0.921749305250841)
rulelist.append(FST.re("$^rewrite((th):θ)"))
#total = 5383 (0.787333625859295)
rulelist.append(FST.re("$^rewrite((dd):ð)"))
#total = 4019 (0.5878309199941495)
rulelist.append(FST.re("$^rewrite((ph):f)"))
#total = 3873 (0.5664765247915753)
rulelist.append(FST.re("$^rewrite((ll):ɬ)"))
#total = 2982 (0.4361562088635366)
rulelist.append(FST.re("$^rewrite((rr):r)"))
#total = 2946 (0.43089074155331286)
rulelist.append(FST.re("$^rewrite((rh):(r̥))"))
#total = 2844 (0.4159719175076788)
rulelist.append(FST.re(
	"$^rewrite('':ː / # $c* [ie] _ #)",
	{'c':wf.sylminus}
))
#total = 2794 (0.4086587684657013)
rulelist.append(FST.re("$^rewrite(t:d / ɬ _)"))
#total = 2776 (0.40602603481058946)

#make rule dictionary
rules = {}
for i in range(len(rulelist)):
	key = 'r' + str(i + 1)
	rules[key] = rulelist[i]

#make the rule string
cs = ['$' + k for k in rules.keys()]
rulestring = ' @ '.join(cs)

#compose all the rules
everything = FST.re(rulestring,rules)

#randomize
random.shuffle(lines)

#go through all items
total = 0
for line in lines:
	word,trans = line.split('\t')
	word = word.lower()
	trans = ''.join(trans.split())
	#get current output
	output = list(everything.generate(word))[0]
	#measure distance from correct outpu
	d = editdistance.eval(trans,output)
	total += d
	#display
	print(f'{word} -> {output}, ({trans} - {d})')

#mismatches and mismatches divided by number of items
print(f'total = {total} ({total/len(lines)})')

