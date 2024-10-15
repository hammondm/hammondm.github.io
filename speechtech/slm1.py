
from pyfoma import FST

x = FST.re("(a:a<1>)*(a:a<3>|a:b<1>)")

print(list(x.apply("aa",weights=True)))

