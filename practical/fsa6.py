from pyfoma import FST

a = FST.re("~(a*)")

print(list(a.generate('aa')))
print(list(a.generate('aab')))
print(list(a.generate('bcd')))

