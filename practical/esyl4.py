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

#affricates
aff = FST.re("ts|tʃ|d̥ʒ̊|d̥z̥|tsʲ")

#onsets
onset = FST.re(
	"s? (($sm? [rl]?)|$aff)",
	{'sm':ef.sylminus,'aff':aff}
)

off = "u̯"[1]

#codas
coda = FST.re(
	"(m|h|r|n|l|nʲ|lʲ|" + off + ")? ː? ($sm|$aff){0,1} (s|t|ɡ̊|k|sː|m|v|n)?",
	{'sm':ef.sylminus,'aff':aff}
)

#simple syllables
syllable = FST.re(
	"$o $sp $c",
	{'o':onset,'sp':ef.sylplus,'c':coda}
)

#words are one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions without vowels
exceptions = FST.re("k|b̥|s|ts|t|tː|v|ue̯i̯ls|pøi̯ɑ̯l")

everything = FST.re(
   "$d @ $s @ ($w|$e)",
   {'d':drop,'s':spaces,'w':word,'e':exceptions}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

