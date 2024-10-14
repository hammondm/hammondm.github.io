from pyfoma import FST

a = FST.re("a(b|c)*de")

#legal
print(list(a.generate("abde")))
#illegal
print(list(a.generate("bde")))

