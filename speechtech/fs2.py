from pyfoma import FST

a = FST.re("$^rewrite(a:A / _ n)")

print(list(a.generate("anata")))

