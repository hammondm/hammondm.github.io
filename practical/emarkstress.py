from pyfoma import FST
import efeatures as ef

f = open('estress.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#transcription fixes
fix0 = FST.re("$^rewrite(ʒ:(ʒ̊) / _ (' '|#))")
fix1 = FST.re("$^rewrite((lːʲ):(lʲː))")
fix2 = FST.re("$^rewrite((sːʲ):(sʲː))")
fix3 = FST.re("$^rewrite((o̯̯):(o̯))")
fix4 = FST.re("$^rewrite(ʋ:v)")
fix5 = FST.re("$^rewrite((ʒ̥):(ʒ̊))")

#push letters together
spaces = FST.re("$^rewrite(('.'|' '):'')")

#affricates
aff = FST.re("ts|tʃ|d̥ʒ̊|d̥z̥|tsʲ")

#onsets
onset = FST.re(
   "([sʃ]? (($sm? [rlvn]?)|$aff) j?)|(ps)",
   {'sm':ef.sylminus,'aff':aff}
)

#get the offglide diacritic
off = "u̯"[1]

#codas
coda = FST.re(
   "(k|m|h|r|n|l|nʲ|lʲ|" + off + ")? ː? " +
	"($sm|$aff){0,1} (s|t|ɡ̊|k|sː|m|v|n)?",
   {'sm':ef.sylminus,'aff':aff}
)

#simple syllables
syllable = FST.re(
	"((ˈ $o $sp $c):s) | " +
	"(($o ˈ $sp $c):s) | " +
	"((ˌ $o $sp $c):2) | " +
	"(($o ˌ $sp $c):2) | " +
	"(($o $sp $c):w)",
	{'o':onset,'sp':ef.sylplus,'c':coda}
)

#words are one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions
exceptions = FST.re(
	#"ˈiɡrek|v|tː|t|ˈb̥|ˈb̥|ˈts|ˈk|ˈs|yhisˌkˈond̥|ˈsɤnʲˑik"
	"v|tː|t|ˈb̥|ˈb̥|ˈts|ˈk|ˈs"
)

everything = FST.re(
	"$d @ $f0 @ $f1 @ $f2 @ $f3 @ $f4 @ $f5 @ $s @ ($w|$e)",
	{'d':drop,'s':spaces,'w':word,'e':exceptions,
	'f0':fix0,'f1':fix1,'f2':fix2,'f3':fix3,'f4':fix4,
	'f5':fix5}
)

for line in lines:
	res = set(everything.generate(line))
	if len(res) == 0:
		print(f'{line}: {res}')

