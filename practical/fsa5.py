from pyfoma import FST

a1 = FST.re("ab*")
a2 = FST.re("a*b")
abx = FST.re("$a1 & $a2",{'a1':a1,'a2':a2})

print(list(abx.generate('aabb')))
print(list(abx.generate('abb')))
print(list(abx.generate('aab')))
print(list(abx.generate('ab')))

