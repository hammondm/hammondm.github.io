from pyfoma import FST
import kfeatures as kf

f = open('kir_cyrl_broad.tsv','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#simple syllables
syllable = FST.re(
	"$sm* $sp $sm*",
	{'sp':kf.sylplus,'sm':kf.sylminus}
)

#words are one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#fix transcription
orule = FST.re("$^rewrite((ʲo):(j o))")
arule = FST.re("$^rewrite((ǎ):ɑ)")

#exceptions without vowels
exceptions = FST.re("k|q|ŋ|b|w|v|ɡ|ɢ|ʁ|d|d͡ʒ|ʒ")

everything = FST.re(
	"$d @ $o @ $a @ $s @ ($w|$e)",
	{'d':drop,'o':orule,'a':arule,
	's':spaces,'w':word,'e':exceptions}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

