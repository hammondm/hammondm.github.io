#import re
from pyfoma import FST
import wfeatures as fw

#stress in Welsh with pyfoma

f = open('wikicym2.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#get rid of marked syllable breaks
breaks = FST.re("$^rewrite('.':'')")

#onset from wsyl4.py plus [r̊]
onset = FST.re(
	"(s? $sm{0,2})|([ɡŋ] w [lrn])|r̊",
	{'sm':fw.sylminus}
)

#syllable plus [ɨ̞], [i̞], and [kᵊ]
syllable = FST.re(
	"($o ($sp|ɨ̞) ($sm{0,3}|i̞))|kᵊ",
	{'sp':fw.sylplus,'sm':fw.sylminus,'o':onset}
)

#word is syllables plus stress marks
word = FST.re(
	"[ˌˈ]? ($s):w ([ˌˈ]? ($s):w)*",
	{'s':syllable}
)

#mark primary stress
stress = FST.re(
	"$^rewrite((ˈw):s)"
)

#mark secondary stress
secondary = FST.re(
	"$^rewrite((ˌw):2)"
)

#patterns to search for
pat = FST.re(
	".*"
)

#put it all together
everything = FST.re(
	"$d @ $s @ $b @ $w @ $str @ $sec @ $p",
	{'d':drop,'s':spaces,'b':breaks,
	'w':word,'str':stress,'sec':secondary,
	'p':pat}
)

for line in lines:
	res = set(everything.generate(line))
	if len(res) > 0:
		print(f'{line} --- {res}')

