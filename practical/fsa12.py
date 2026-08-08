from pyfoma import FST

a = FST.re("$^rewrite('':a / _ b)")

print(list(a.generate('b')))
print(list(a.generate('c')))
print(list(a.generate('bbb')))

