from pyfoma import FST
import re

#file in unimorph format
filename = 'wikifasortho.txt'

#get all the tags from a phrase
def gettags(x):
	res = set()
	words = x.split()
	for word in words:
		tags = re.sub('^[^;]*','',word)
		tags = tags.split(';')
		for tag in tags:
			if len(tag) > 0: res.add(tag)
	return res

#read in data
f = open(filename,'r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

stem = FST.re("""
(krdn):(krd)|
(grftn):(grft)|
(aistadn):(aistad)|
(Sdn):(Sd)
""")

stem2 = FST.re("""
(krdn):(kn)|
(grftn):(gir)|
(aistadn):(aist)|
(Sdn):(Su?)
""")

nve = FST.re("""
a':'btni|
atraq|
aJra|
aHsas|
adarh|
iad|
a':'raiS|
a':'r(a|u)m|
azduaJ|
astraHt|
astfadh|
astmna|
aski|
aStbah|
a':'Sna|
as'*'rar|
az'+'afh|
a'?''*'traz'+'|
a'?''*'lam
""")

#1st conjugation
pn1sg = FST.re("((1 ; 'SG' (; 'COL')?):m)")
pn1pl = FST.re("((1 ; 'PL' (; 'COL')?):(im))")
pn2sg = FST.re("((2 ; 'SG' (; 'COL')?):(i))")
pn2pl = FST.re(
	"((2 ; 'PL' ; 'COL'):(in))|((2 ; 'PL'):(id))"
)
pn3sg = FST.re("((3 ; 'SG' (; 'COL')?):(''|h|d))")
pn3pl = FST.re(
	"((3 ; 'PL'):(nd))|((3 ; 'PL' ; 'COL'):n)"
)

#1st conjugation
pn = FST.re(
	"$pn1sg|$pn1pl|$pn2sg|$pn2pl|$pn3sg|$pn3pl",
	{'pn1sg':pn1sg,'pn1pl':pn1pl,'pn2sg':pn2sg,
	'pn2pl':pn2pl,'pn3sg':pn3sg,'pn3pl':pn3pl}
)

#2nd conjugation
pf1sg = FST.re("((1 ; 'SG'):m)")
pf1pl = FST.re("((1 ; 'PL'):(im))")
pf2sg = FST.re("((2 ; 'SG'):(i))")
pf2pl = FST.re("((2 ; 'PL'):(id))")
pf3sg = FST.re("((3 ; 'SG'):d)")
pf3pl = FST.re("((3 ; 'PL'):(nd))")

#2nd conjugation
pf = FST.re(
	"$pf1sg|$pf1pl|$pf2sg|$pf2pl|$pf3sg| $pf3pl",
	{'pf1sg':pf1sg,'pf1pl':pf1pl,'pf2sg':pf2sg,
	'pf2pl':pf2pl,'pf3sg':pf3sg,'pf3pl':pf3pl}
)

#past tense
past = FST.re(
	"($nve ' ')? $stem ((; V ;):'') $pn ((; 'PST'):'')",
	{'nve':nve,'stem':stem,'pn':pn}
)

#future tense
fut = FST.re(
	"($nve ' ')? xuah (;:'') $pf ' ' $stem " + \
	"((; V ; 'FUT'):'')",
	{'nve':nve,'stem':stem,'pf':pf}
)

#present subjunctive
subj = FST.re(
	"($nve ' ')? '':b $stem2 ((; V ; 'SBJV' ; " + \
	"'PRS' ;):'') $pn",
	{'nve':nve,'stem2':stem2,'pn':pn}
)

#imperative person marking
pi2pl = FST.re(
	"((2 ; 'PL' ; 'COL'):(in))|((2 ; 'PL'):(id))"
)
pi2sg = FST.re("((2 ; 'SG' (; 'COL')?):'')")

#imperative
imp = FST.re(
	"($nve ' ')? '':b $stem2 (; V ; 'IMP' ;):''" + \
	" ($pi2pl|$pi2sg)",
	{'nve':nve,'stem2':stem2,'pi2pl':pi2pl,
	'pi2sg':pi2sg}
)

#present imperfective
pipfv = FST.re(
	"($nve ' ')? '':(mi) $stem2 (; V ; 'PRS' ; " + \
	"'IPFV' ;):'' $pn",
	{'nve':nve,'stem2':stem2,'pn':pn}
)

#past progressive
ppst = FST.re(
	"'':(daSt) ;:'' $pn ' ' ($nve ' ')? '':(mi) $stem" + \
	" (; V ; 'PST' ; 'PROG' ;):'' $pn",
	{'nve':nve,'stem':stem,'pn':pn}
)

#present progressive
prsprg = FST.re(
	"'':(dar) ;:'' $pn ' ' ($nve ' ')? '':(mi) $stem2" + \
	" (; V ; 'PRS' ; 'PROG' ;):'' $pn",
	{'nve':nve,'stem2':stem2,'pn':pn}
)

#participles
ptcp = FST.re(
	"($nve ' ')? (($stem2 ((; 'PRS'):(ndh)))|($stem" + \
	" ((; 'PST'):h)))" + \
	" ((; V ;):'') (('V.PTCP' (; 'COL')?):'')",
	{'nve':nve,'stem':stem,'stem2':stem2}
)

#past imperfective
pstipf = FST.re(
	"($nve ' ')? '':(mi) $stem" + \
	" (; V ; 'PST' ; 'IPFV' ;):'' $pn",
	{'nve':nve,'stem':stem,'pn':pn}
)

#perfective
pfv = FST.re(
	"($nve ' ')? $stem2 (; V ; 'PFV' ;):'' $pn",
	{'nve':nve,'stem2':stem2,'pn':pn}
)

#past perfect
pstprf = FST.re(
	"($nve ' ')? $stem '':h (; V ; 'PST' ; 'PRF' ):'' " + \
	"' ' ';':(bud) $pn",
	{'nve':nve,'stem':stem,'pn':pn}
)

#past subjunctive
pstsbj = FST.re(
	"($nve ' ')? $stem '':h (; V ; 'PST' ; 'SBJV'):'' " + \
	"' ' ';':(baS) $pn",
	{'nve':nve,'stem':stem,'pn':pn}
)

#3rd conjugation
pp1sg = FST.re("((1 ; 'SG' (; 'COL')?):(am))")
pp1pl = FST.re("((1 ; 'PL' (; 'COL')?):(aim))")
pp2sg = FST.re("((2 ; 'SG' (; 'COL')?):(ai))")
pp2pl = FST.re(
	"((2 ; 'PL' ; 'COL'):(ain))|((2 ; 'PL'):(aid))"
)
pp3sg = FST.re(
	"((3 ; 'SG' ; 'COL'):'')|((3 ; 'SG'):(' ' ast))"
)
pp3pl = FST.re(
	"((3 ; 'PL'):(and))|((3 ; 'PL' ; 'COL'):(an))"
)
#3rd conjugation
pp = FST.re(
	"$pp1sg|$pp1pl|$pp2sg|$pp2pl|$pp3sg| $pp3pl",
	{'pp1sg':pp1sg,'pp1pl':pp1pl,'pp2sg':pp2sg,
	'pp2pl':pp2pl,'pp3sg':pp3sg,'pp3pl':pp3pl}
)

#nonfinite
nfin = FST.re(
	"($nve ' ')? $stem n (; V ; 'NFIN' (; 'COL')?):''",
	{'nve':nve,'stem':stem}
)

#present perfect
ppf = FST.re(
	"($nve ' ')? $stem h (; V ; 'PRS' ; 'PRF' ;):'' $pp",
	{'nve':nve,'stem':stem,'pp':pp}
)

#everything
phrase = FST.re(
	"$past|$fut|$subj|$imp|$pipfv|$ppst|" + \
	"$prsprg|$pstipf|$pfv|$pstprf|$pstsbj|" + \
	"$ppf|$ptcp|$nfin",
	{'past':past,'fut':fut,'subj':subj,
	'prsprg':prsprg,'imp':imp,'pipfv':pipfv,
	'ppst':ppst,'pstipf':pstipf,'pfv':pfv,
	'pstprf':pstprf,'pstsbj':pstsbj,
	'ppf':ppf,'ptcp':ptcp,'nfin':nfin}
)

total = 0
good = 0
for line in lines[:2720]:
	total += 1
	lemma,form,tags = line.split('\t')
	print(line)
	itags = set(tags.split(';'))
	res = list(phrase.analyze(form))
	#print(f'\t{res}')
	for frm in res:
		otags = gettags(frm)
		if itags == otags:
			print(f'\t\t{frm}')
			good += 1
			break

print(f'{good}/{total} = {good/total:.2f}')

