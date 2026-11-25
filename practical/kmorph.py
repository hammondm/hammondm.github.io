from pyfoma import FST
import re

f = open('kortho.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#get rid of tags at the end
drop = FST.re("$^rewrite(('-'.*):'',longest=True)")

#function to create rules
def makerule(a,b,c,d):
	res = "$^rewrite(" + a + ":" + b + " / "
	res += (c + " _ " + d + ")")
	return FST.re(res)

#all the rules
rulelists = [
	#progressive form
	#	irregulars
	["(uu)","(Ip ' ' Zat)","# koS","'-' V ';' PROG"],
	["(uu)","(uup ' ' Zat)","# Z","'-' V ';' PROG"],
	#	regulars
	["(OO)","(Op ' ' Zat)","","'-' V ';' PROG"],
	["(UU)","(Up ' ' Zat)","[OU]..?","'-' V ';' PROG"],
	["(UU)","(ip ' ' Zat)","[ie]..?","'-' V ';' PROG"],
	["(oo)","(ap ' ' Zat)","[Iua]..?","'-' V ';' PROG"],
	["(oo)","(op ' ' Zat)","o..?","'-' V ';' PROG"],
	["(uu)","(Ip ' ' Zat)","[Ia]..?","'-' V ';' PROG"],
	["(uu)","(up ' ' Zat)","[ou]..?","'-' V ';' PROG"],
	#1st sg
	["''","(damIn)","(uu|t)","'-' V (';' PROG)? ';' SG ';' 1 ';' PRS"],
	["''","(domun)","oo","'-' V ';' SG ';' 1 ';' PRS"],
	["''","(dOmUn)","(UU|OO)","'-' V ';' SG ';' 1 ';' PRS"],
	#1st pl
	["''","(dabIz)","(uu|t)","'-' V (';' PROG)? ';' PL ';' 1 ';' PRS"],
	["''","(dobuz)","oo","'-' V ';' PL ';' 1 ';' PRS"],
	["''","(dObUz)","(UU|OO)","'-' V ';' PL ';' 1 ';' PRS"],
	#3rd
	["''","(da)","uu","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(do)","oo","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(dO)","(UU|OO)","'-' V ';' (SG|PL) ';' 3 ';' PRS"],
	["''","(at)","t","'-' V ';' PROG ';' SG ';' 3 ';' PRS"],
	["''","(ISat)","t","'-' V ';' PROG ';' PL ';' 3 ';' PRS"],
	#2nd pl informal
	["''","(dasINar)","(uu|t)",
		"'-' V (';' PROG)? ';' PL ';' 2 ';' INFM ';' PRS"],
	["''","(dosuNar)","oo","'-' V ';' PL ';' 2 ';' INFM ';' PRS"],
	["''","(dOsUNOr)","(UU|OO)","'-' V ';' PL ';' 2 ';' INFM ';' PRS"],
	#2nd sg informal
	["''","(dasIN)","(uu|t)",
		"'-' V (';' PROG)? ';' SG ';' 2 ';' INFM ';' PRS"],
	["''","(dosuN)","oo","'-' V ';' SG ';' 2 ';' INFM ';' PRS"],
	["''","(dOsUN)","(UU|OO)","'-' V ';' SG ';' 2 ';' INFM ';' PRS"],
	#2nd sg formal
	["''","(dasIz)","(uu|t)",
		"'-' V (';' PROG)? ';' FRML ';' SG ';' 2 ';' PRS"],
	["''","(dosuz)","oo","'-' V ';' FRML ';' SG ';' 2 ';' PRS"],
	["''","(dOsUz)","(UU|OO)","'-' V ';' FRML ';' SG ';' 2 ';' PRS"],
	#fix progressives
	["d","''","Zat",""],
	#fix baStap,ZazIp,ZumuS kIlIp,karmap,tamaktanIp
	["(In)","''",
		"(baStap|ZazIp|ZumuS ' ' kIlIp|karmap|tamaktanIp) ' ' Zatam",""],
	#fix kUnOOlOO
	["(OO)","(UU)","kUnOOl",""]
]

#create the rules
rules = {}
names = []
count = 0
for (a,b,c,d) in rulelists:
	count += 1
	name = 'r' + str(count)
	names.append(name)
	rules[name] = makerule(a,b,c,d)

#add the drop rule
names.append('d')
rules['d'] = drop

#make the rule string
s = ' @ '.join(['$' + name for name in names])

#compose all the rules
everything = FST.re(s,rules)

#go through all the lines
items = 0
success = 0
for line in lines:
	#only look at the present tense!
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

