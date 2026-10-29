from pyfoma import FST
import kfeatures as kf

f = open('kstress.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

onset = FST.re(
	"$sm? | ($obs [rl]) | (st)",
	{'sm':kf.sylminus,'obs':kf.sonminus}
)

coda = FST.re(
	"[vɫŋrln]? $sm?",
	{'sm':kf.sylminus}
)

#simple syllables
syllable = FST.re(
	"$o $sp $c",
	{'o':onset,'sp':kf.sylplus,'c':coda}
)

#words are one or more syllables
word = FST.re(
	#"$s+",
	"[ˌˈ]? ($s):w ([ˌˈ]? ($s):w)*",
	{'s':syllable}
)

#fix transcription
orule = FST.re("$^rewrite((ʲo):(j o))")
arule = FST.re("$^rewrite((ǎ):ɑ)")
zhrule = FST.re("$^rewrite((d ' ' ʒ):(d͡ʒ))")
chrule = FST.re("$^rewrite((t ' ' ʃ):(t͡ʃ))")

#exceptions without vowels
exceptions = FST.re("k|q|ŋ|b|w|v|ɡ|ɢ|ʁ|d|d͡ʒ|ʒ")

#mark primary stress
stress = FST.re(
   "$^rewrite((ˈw):s)"
)

#mark secondary stress
secondary = FST.re(
   "$^rewrite((ˌw):2)"
)

everything = FST.re(
	#"$d @ $o @ $a @ $z @ $c @ $s @ ($w|$e) @ $st @ $sec",
	"$d @ $o @ $a @ $z @ $c @ $s @ $w @ $st @ $sec",
	{'d':drop,'o':orule,'a':arule,
	'z':zhrule,'c':chrule,'s':spaces,
	'w':word,'e':exceptions,'st':stress,
	'sec':secondary}
)

for line in lines:
	res = set(everything.generate(line))
	if len(res) > 0:
		print(f'{line}: {res}')

