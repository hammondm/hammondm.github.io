from pyfoma import FST

vowel = FST.re("(a|i|u)")

nas = FST.re("(m|n)")

vnas = FST.re("(a'~'|i'~'|u'~')")

rule = FST.re(
	"$^rewrite(($v:$vn) / _ $n)",
	{'v':vowel,'n':nas,'vn':vnas}
)

print(list(rule.generate("pan")))
print(list(rule.generate("pat")))
print(list(rule.generate("pin")))
print(list(rule.generate("pit")))

