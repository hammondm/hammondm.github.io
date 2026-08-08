from pyfoma import FST

a = FST.re("a|(a:c)")

print(list(a.generate('a')))
print(list(a.generate('b')))
print(list(a.analyze('a')))
print(list(a.analyze('c')))

