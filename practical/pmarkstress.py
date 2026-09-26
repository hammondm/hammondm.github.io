from pyfoma import FST
import pfeatures as pf

#read in data
f = open('wikifasSTRESS.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#fix [g] problem
grule = FST.re("$^rewrite(g:ɡ)")

#eliminate syllable boundaries and stress marks
dotrule = FST.re("$^rewrite(('.'|ˈ):'')")

#fix t͡ʃ problem
chrule = FST.re("$^rewrite((tʃ):(t͡ʃ))")

#onset rule
onset = FST.re(
	"$sm? | $sm [mvrlɾw]",
	{'sm':pf.sylminus}
)

#simple syllables
syllable = FST.re(
	"$o $sp $sm{0,2}",
	{'sp':pf.sylplus,'sm':pf.sylminus,'o':onset}
)

#a word is one or more syllables
word = FST.re(
	"$s ('':'|' $s)*",
	{'s':syllable}
)

#max onset
mo = FST.re(
	"~(.* $c '|' $v .*)",
	{'c':pf.sylminus,'v':pf.sylplus}
)

everything = FST.re(
	"$d @ $s @ $g @ $dot @ $ch @ $w @ $mo",
	{'d':drop,'s':spaces,'g':grule,'dot':dotrule,
	'ch':chrule,'w':word,'mo':mo}
)

#eliminate syllable boundaries
justdotrule = FST.re("$^rewrite('.':'')")

targ = FST.re(
	"$d @ $s @ $g @ $j",
	{'d':drop,'s':spaces,'g':grule,
	'j':justdotrule}
)

s1 = FST.re(
	"$^rewrite('':ˈ / ('|'|#) _ [^'|']* #)"
)

clean = FST.re(
	"$^rewrite('|':'')"
)

stress = FST.re(
	"$s1 @ $clean",
	{'s1':s1,'clean':clean}
)

success = 0
#for line in lines[:100]:
for line in lines:
	res = set(everything.generate(line))
	target = list(targ.generate(line))[0]
	#print(f'{line} -> {target}')
	routs = []
	for r in res:
		rout = list(stress.generate(r))[0]
		routs.append(rout)
		#print(f'\t{r} -> {rout}')
	if target in routs:
		success += 1

print(f'success = {success/len(lines):.2f}')

