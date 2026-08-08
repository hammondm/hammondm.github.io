from pyfoma import FST

a = FST.re('(ab)*')

print(a.view(show_alphabet=False))

