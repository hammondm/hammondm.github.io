from pyfoma import FST
import wfeatures as wf

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

#get arcs and states
print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

