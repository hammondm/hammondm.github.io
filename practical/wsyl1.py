from pyfoma import FST
import wfeatures as wf

f = open('wikicym3.txt','r')
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
	{'sp':wf.sylplus,'sm':wf.sylminus}
)

word = FST.re(
	"$s+",
	{'s':syllable}
)

everything = FST.re(
	"$d @ $s @ $w",
	{'d':drop,'s':spaces,'w':word}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

