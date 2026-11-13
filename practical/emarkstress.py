from pyfoma import FST
import efeatures as ef

f = open('estress.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#long vowels
lv = FST.re(
	"$sp & ($long|$half)",
	{'sp':ef.sylplus,'long':ef.longplus,'half':ef.halflongplus}
)

#short vowels
sv = FST.re(
	"$sp - $lv",
	{'sp':ef.sylplus,'lv':lv}
)

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#transcription fixes
fix0 = FST.re("$^rewrite(ʒ:(ʒ̊) / _ (' '|#))")
fix1 = FST.re("$^rewrite((lːʲ):(lʲː))")
fix2 = FST.re("$^rewrite((sːʲ):(sʲː))")
fix3 = FST.re("$^rewrite((o̯̯):(o̯))")
fix4 = FST.re("$^rewrite(ʋ:v)")
fix5 = FST.re("$^rewrite((ʒ̥):(ʒ̊))")
fix6 = FST.re("$^rewrite(ˌ:'' / yhis _ kˈond̥)")
fix7 = FST.re("$^rewrite(.:(ɡ̊) / ˈi _ rek)")
fix8 = FST.re("$^rewrite((ɑ̯):ɑ / ˈpøi̯ _ l)")

#push letters together
spaces = FST.re("$^rewrite(('.'|' '):'')")

#add [nʲˑ] to consonants
sylm = FST.re("$sm|(nʲˑ)",{'sm':ef.sylminus})

#glides
glide = FST.re(
	"$sylm & $conm",
	{'sylm':ef.sylminus,'conm':ef.consminus}
)

#affricates
aff = FST.re("ts|tʃ|d̥ʒ̊|d̥z̥|tsʲ")

#onsets
onset = FST.re(
	"([sʃ]? (($sm? [rlvn]?)|$aff) j?)|(ps)",
	{'sm':sylm,'aff':aff}
)

#left side of coda
c1 = FST.re(
	#"((k|m|h|r|n|l|nʲ|lʲ) ː?)? ",
	"(((k|m|h|r|n|l|nʲ|lʲ) ː?)|$g)? ",
	{'g':glide}
)

#right side of coda
c2 = FST.re(
	#"($sm|$aff){0,1} (s|t|ɡ̊|k|sː|m|v|n)?",
	"($sm|$aff) (s|t|ɡ̊|k|sː|m|v|n)?",
	{'sm':sylm,'aff':aff}
)

#codas
coda = FST.re(
	"$c1 $c2",
	{'c1':c1,'c2':c2}
)

#nucleus
nuc = FST.re(
	#"$sp $g?",
	"$sp",
	{'sp':ef.sylplus,'g':glide}
)

lightsyl = FST.re(
	"$o $sv",
	{'o':onset,'sv':sv}
)

heavysyl = FST.re(
	"$o ($lv|($n $c))",
	{'o':onset,'lv':lv,'n':nuc,'c':coda}
)

#simple syllables
syllable = FST.re(
	#"((ˈ $o $n $c):s) | " +
	"((ˈ $hs):S) | " +
	"((ˈ $ls):s) | " +
	#"(($o ˈ $n $c):s) | " +
	"(($o ˈ ($lv|($n $c))):S) | " +
	"(($o ˈ $sv):s) | " +
	#"((ˌ $o $n $c):2) | " +
	"((ˌ $ls):2) | " +
	"((ˌ $hs):3) | " +
	#"(($o ˌ $n $c):2) | " +
	"(($o ˌ ($lv|($n $c))):3) | " +
	"(($o ˌ $sv):2) | " +
	#"(($o $n $c):w)",
	"(($ls):w) | " +
	"(($hs):W)",
	{'o':onset,'n':nuc,'c':coda,
	'hs':heavysyl,'ls':lightsyl,
	'lv':lv,'sv':sv}
)

#words are one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions
exceptions = FST.re(
	"v|tː|t|ˈb̥|ˈb̥|ˈts|ˈk|ˈs|ˈue̯i̯ls"
)

mark = FST.re(
	"([wW]+) | " +
	"([23Ss].*) | " +
	"([wW]+[3S].*)"
)

everything = FST.re(
	"$d @ $f0 @ $f1 @ $f2 @ $f3 @ " +
	"$f4 @ $f5 @ $s @ $f6 @ $f7 @ $f8 @ " +
	"(($w @ $m)|$e)",
	#"($w|$e)",
	{'d':drop,'s':spaces,'w':word,'e':exceptions,
	'f0':fix0,'f1':fix1,'f2':fix2,'f3':fix3,'f4':fix4,
	'f5':fix5,'f6':fix6,'f7':fix7,'f8':fix8,'m':mark}
)

for line in lines:
	res = set(everything.generate(line))
	if len(res) == 0:
	#if len(res) > 0:
		print(f'{line}: {res}')

