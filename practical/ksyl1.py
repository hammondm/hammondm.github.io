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

everything = FST.re(
	"$d @ $o @ $a @ $s @ $w",
	{'d':drop,'o':orule,'a':arule,
	's':spaces,'w':word}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

