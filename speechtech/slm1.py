import pywrapfst as fst

letters = ['<epsilon>','a','b']
st = fst.SymbolTable()
for letter in letters:
    st.add_symbol(letter)

c = fst.Compiler(
    isymbols=st,
    osymbols=st,
    keep_isymbols=True,
    keep_osymbols=True
)

c.write('0 0 a a 1')
c.write('0 1 a a 3')
c.write('0 2 a b 1')
c.write('1')
c.write('2')
f = c.compile()

c2 = fst.Compiler(
    isymbols=st,
    osymbols=st,
    keep_isymbols=True,
    keep_osymbols=True
)

c2.write('0 1 a a')
c2.write('1')
f2 = c2.compile()

f3 = fst.compose(f2,f)

print(f'f\n{f}')
print(f'f3\n{f3}')

print(fst.shortestpath(f3))

