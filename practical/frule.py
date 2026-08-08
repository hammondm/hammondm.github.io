from pyfoma import FST

nas = FST.re("(m|n)")

arule = FST.re("$^rewrite(a:(a'~') / _ $n)",{'n':nas})
irule = FST.re("$^rewrite(i:(i'~') / _ $n)",{'n':nas})
urule = FST.re("$^rewrite(u:(u'~') / _ $n)",{'n':nas})

rule = FST.re(
	"$arule @ $irule @ $urule",
	{'arule':arule,'irule':irule,'urule':urule}
)

print(list(rule.generate("pan")))
print(list(rule.generate("pat")))
print(list(rule.generate("pin")))
print(list(rule.generate("pit")))

