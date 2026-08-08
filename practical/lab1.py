from pyfoma import FST

lab = FST.re("(b|p|v|w|f)")

con = FST.re("~(.* f $lab .*)",{'lab':lab})

print(list(con.generate("afta")))
print(list(con.generate("afba")))
print(list(con.generate("afpa")))
print(list(con.generate("afva")))
print(list(con.generate("afwa")))
print(list(con.generate("affa")))

