from pyfoma import FST
import wfeatures as fw

#read in data
f = open('wikicym2.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#transcription fixes
fix1 = FST.re("$^rewrite((t ' ' ʃ):(t͡ʃ))")
fix2 = FST.re("$^rewrite((kᵊ):(k ə))")
fix3 = FST.re("$^rewrite((d ' ' ʒ):(d͡ʒ))")

#drop stresses
drop = FST.re("$^rewrite(('.'|ˌ|ˈ):'')")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#consonants
c = FST.re(
	"$sm|r̊|i̞",
	{'sm':fw.sylminus}
)

#vowels
v = FST.re(
	"$sp|ɨ̞",
	{'sp':fw.sylplus}
)

#onsets
onset = FST.re(
	"(s? (($c & $sm)|r̊|$np|h)? ($c & $sp)?)|([ɡŋ]? w [lrn])",
	{'c':c,'sm':fw.sonminus,'sp':fw.sonplus,'np':fw.nasplus}
)

#codas
coda = FST.re(
	"(s|($g & $c))? $c{0,2}",
	{'g':fw.sonplus,'c':c}
)

#syllables
syllable = FST.re(
	"$o $v $c",
	{'v':v,'c':coda,'o':onset}
)

#words
word = FST.re(
	"$s (('':'|') $s)*",
	{'s':syllable}
)

#at least one consonant is an onset
mo = FST.re(
	"~(.* $c '|' $v .*)",
	{'c':c,'v':v}
)

#stress a final long vowel
sf = FST.re(
	"$^rewrite('':ˈ / '|' _  [^'|']* ː [^'|']* #)"
)

#primary stress
s1 = FST.re(
	"$^rewrite('':ˈ / (#|'|') _ [^'|'|ˈ]* '|' [^'|'|ˈ]* #)"
)

#secondary stress
s2 = FST.re(
	"$^rewrite('':ˌ / (#|'|') _ [^'|']* '|' [^'|']* '|' ˈ)"
)

#remove syllable boundaries
clean = FST.re("$^rewrite('|':'')")

#put it all together (without stress)
everything = FST.re(
	"$f1 @ $f2 @ $f3 @ $d @ $s @ $w @ $mo",
	{'f1':fix1,'f2':fix2,'f3':fix3,
	'd':drop,'s':spaces,'w':word,'mo':mo}
)

#mark stress
stress = FST.re(
	"$sf @ $s1 @ $s2 @ $cl",
	{'sf':sf,'s1':s1,'s2':s2,'cl':clean}
)

#fix transcription
fix = FST.re(
	"$f1 @ $f2 @ $f3 @ $s @ $^rewrite('.':'')",
	{'f1':fix1,'f2':fix2,'f3':fix3,
	's':spaces}
)

#go through all items
count = 0
for line in lines:
	word,trans = line.split('\t')
	#syllabify
	res = list(everything.generate(trans))
	#fix input
	fres = list(fix.generate(trans))[0]
	reslist = []
	for r in res:
		#add stress to each syllabified form
		sres = list(stress.generate(r))[0]
		reslist.append(sres)
	#allow for mis-syllabified items
	if fres in reslist:
		count += 1
	#print mis-stressed items
	else:
		print(line)
		for r1,r2 in zip(res,reslist):
			print(f'\t{r1} -> {r2} ({fres})')

print(f'percent correct: {count/len(lines):.2f}')

