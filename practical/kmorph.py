from pyfoma import FST
import re

f = open('kortho.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#get rid of tags at the end
drop = FST.re("$^rewrite(('-'.*):'',longest=True)")

def makerule(a,b,c,d):
	res = "$^rewrite(" + a + ":" + b + " / "
	res += (c + " _ " + d + ")")
	return FST.re(res)

#berip

rulelists = [
	#make progressive form
	["(uu)","(Ip ' ' Zat)","(aCuulan|aC)","'-' V ';' PROG"],
	["(oo)","(op ' ' Zat)","ojn","'-' V ';' PROG"],
	["(oo|uu|UU)","(Ip ' ' Zat)","(I.|a[Srl])","'-' V ';' PROG"],
	["(uu)","(up ' ' Zat)","[ou].","'-' V ';' PROG"],
	["(oo|uu|UU)","(ap ' ' Zat)","[Iiua][tsrkSj]?.","'-' V ';' PROG"],
	["(oo|uu|UU)","(Up ' ' Zat)","[OU].","'-' V ';' PROG"],
	["(UU)","(ip ' ' Zat)","ber","'-' V ';' PROG"],
	#1st sg
	["''","(damIn)","(uu|t)","'-' V (';' PROG)? ';' SG ';' 1 ';' PRS"],
	["''","(domun)","oo","'-' V ';' SG ';' 1 ';' PRS"],
	["''","(dOmUn)","UU","'-' V ';' SG ';' 1 ';' PRS"],
	#1st pl
	["''","(dabIz)","(uu|t)","'-' V (';' PROG)? ';' PL ';' 1 ';' PRS"],
	["''","(dobuz)","oo","'-' V ';' PL ';' 1 ';' PRS"],
	["''","(dObUz)","UU","'-' V ';' PL ';' 1 ';' PRS"],
	#3rd
	["''","(da)","uu","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(do)","oo","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(dO)","UU","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(at)","t","'-' V ';' PROG ';' SG ';' 3 ';' PRS"],
	["''","(ISat)","t","'-' V ';' PROG ';' PL ';' 3 ';' PRS"],
	#2nd pl informal
	["''","(dasINar)","(uu|t)",
		"'-' V (';' PROG)? ';' PL ';' 2 ';' INFM ';' PRS"],
	["''","(dosuNar)","oo","'-' V ';' PL ';' 2 ';' INFM ';' PRS"],
	["''","(dOsUNOr)","UU","'-' V ';' PL ';' 2 ';' INFM ';' PRS"],
	#2nd sg informal
	["''","(dasIN)","(uu|t)",
		"'-' V (';' PROG)? ';' SG ';' 2 ';' INFM ';' PRS"],
	["''","(dosuN)","oo","'-' V ';' SG ';' 2 ';' INFM ';' PRS"],
	["''","(dOsUN)","UU","'-' V ';' SG ';' 2 ';' INFM ';' PRS"],
	#2nd sg formal
	["''","(dasIz)","(uu|t)",
		"'-' V (';' PROG)? ';' FRML ';' SG ';' 2 ';' PRS"],
	["''","(dosuz)","oo","'-' V ';' FRML ';' SG ';' 2 ';' PRS"],
	["''","(dOsUz)","UU","'-' V ';' FRML ';' SG ';' 2 ';' PRS"],
	#fix progressives
	["d","''","Zat",""],
	#fix baStap
	["(In)","''","baStap ' ' Zatam",""]
]

rules = {}
names = []
count = 0
for (a,b,c,d) in rulelists:
	count += 1
	name = 'r' + str(count)
	names.append(name)
	rules[name] = makerule(a,b,c,d)
names.append('d')

rules['d'] = drop

s = ' @ '.join(['$' + name for name in names])

everything = FST.re(s,rules)

items = 0
success = 0
#for line in lines[:896]:
for line in lines:
	if re.search('PRS',line):
		items += 1
		lemma,target,tags = line.split('\t')
		inp = lemma + '-' + tags
		outp = list(everything.generate(inp))[0]
		if outp == target:
			success += 1
		else:
			print(f'{inp}: {outp}/{target}')

print(success/items)

