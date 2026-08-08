from pyfoma import FST

a = FST.re("[^ab]")

print(list(a.generate('aa')))
print(list(a.generate('a')))
print(list(a.generate('b')))
print(list(a.generate('c')))
print(list(a.generate('cd')))

