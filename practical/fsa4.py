from pyfoma import FST

a = FST.re('abc')
b = FST.re('def')
c = FST.re('$a|$b',{'a':a,'b':b})

print(c.view(show_alphabet=False))

