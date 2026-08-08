from pyfoma import FST

a = FST.re('(ab)*')

print(list(a.generate('ab')))
print(list(a.generate('aba')))
print(list(a.generate('abab')))

