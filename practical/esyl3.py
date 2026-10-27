from pyfoma import FST
import efeatures as ef

f = open('estfixed.tsv','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#simple syllables
syllable = FST.re(
	"$sm{0,3} $sp $sm{0,3}",
	{'sp':ef.sylplus,'sm':ef.sylminus}
)

#words are one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions without vowels
exceptions = FST.re("k|b̥|s|ts|t|tː|v|ue̯i̯ls")

everything = FST.re(
   "$d @ $s @ ($w|$e)",
   {'d':drop,'s':spaces,'w':word,'e':exceptions}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

