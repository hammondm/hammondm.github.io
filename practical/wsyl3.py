from pyfoma import FST
import wfeatures as fw

f = open('wikicym3.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

syllable = FST.re(
	"$sm{0,3} $sp $sm{0,3}",
	{'sp':fw.sylplus,'sm':fw.sylminus}
)

word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions without vowels
exceptions = FST.re("χ|dw|d|m|h|i̯|m|n|r|θ|i̯|w")

everything = FST.re(
	"$d @ $s @ ($w|$e)",
	{'d':drop,'s':spaces,'w':word,'e':exceptions}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

