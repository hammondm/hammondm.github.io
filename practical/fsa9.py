from pyfoma import FST

a = FST.re("((a:c)|(b:d)) @ (c:e)")

print(list(a.generate('a')))
print(list(a.generate('b')))

