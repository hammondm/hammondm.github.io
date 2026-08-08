from pyfoma import FST

vowel = FST.re("(i|u|a)")

high = FST.re("(i|u|k)")

con = FST.re(
	"$^rewrite((p|t):k / _ ($vowel & $high))",
	{'vowel':vowel,'high':high}
)

print(list(con.generate("pa")))
print(list(con.generate("pi")))
print(list(con.generate("ta")))
print(list(con.generate("ti")))
print(list(con.generate("ka")))
print(list(con.generate("ki")))

