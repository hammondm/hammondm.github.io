from pyfoma import FST

vowel = FST.re("(i|u|a)")

high = FST.re("(i|u|k)")

con = FST.re(
	"~(.* ($vowel & $high))",
	{'vowel':vowel,'high':high}
)

print(list(con.generate("paka")))
print(list(con.generate("pak")))
print(list(con.generate("paku")))
print(list(con.generate("paki")))

