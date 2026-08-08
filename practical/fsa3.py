from pyfoma import FST

a = FST.re('(a|b)*')

print(a.view(show_alphabet=False))

