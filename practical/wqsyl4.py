from pyfoma import FST
import wfeatures as fw

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

onset = FST.re(
	"(s? $sm{0,2})|([ɡŋ] w [lrn])",
	{'sm':fw.sylminus}
)

syllable = FST.re(
	"$o $sp $sm{0,3}",
	{'sp':fw.sylplus,'sm':fw.sylminus,'o':onset}
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

print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

