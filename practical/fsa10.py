from pyfoma import FST

a = FST.re("$^rewrite(a:b / _ c)")

print(list(a.generate('ac')))
print(list(a.generate('ad')))

