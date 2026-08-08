from pyfoma import FST

a = FST.re("$^rewrite(a:'' / _ c)")

print(list(a.generate('abc')))
print(list(a.generate('ac')))
print(list(a.generate('aac')))

