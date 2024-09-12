#mapping from acoustics to phonetics to word tokens

import pywrapfst as fst

#data
words = '''KAT cat
TAK tack
KOT coat
OK oak
OT oat
TO toe
AKT act
KAT cat
TAKT tact'''
words = words.split('\n')

#extract bits for WFSTs
#letters for symbol table and WFST1
letters = set()
#letters and tokens for WFST2
pairs = []
for line in words:
    bits = line.split(' ')
    pairs.append((bits[0].lower(),bits[1]))
    for letter in bits[0]:
        letters.add(letter)

#make symbol table
st = fst.SymbolTable()
st.add_symbol('<epsilon>')
for letter in letters:
    st.add_symbol(letter)
    st.add_symbol(letter.lower())
letters.add('<epsilon>')
for pair in pairs:
    st.add_symbol(pair[1])

#make acoustic model wfst
c = fst.Compiler(
    isymbols=st,
    osymbols=st,
    keep_isymbols=True,
    keep_osymbols=True
)

#arcs for acoustic model wfst
for let1 in letters:
    for let2 in letters:
        if let1 != '<epsilon>' or let2 != '<epsilon>':
            if let1 != let2:
                weight = '1'
            else:
                weight = '0'
            let2 = let2.lower()
            arc = ' '.join(['0','0',let1,let2.lower(),weight])
            c.write(arc)
c.write('0 0')

am = c.compile()

#make language model wfst
c2 = fst.Compiler(
    isymbols=st,
    osymbols=st,
    keep_isymbols=True,
    keep_osymbols=True
)

#make arcs
statenum = 1
finals = []
for pair in pairs:
    lets = pair[0]
    token = pair[1]
    arc = ' '.join(['0',str(statenum),lets[0],token])
    c2.write(arc)
    for let in lets[1:]:
        arc = ' '.join([str(statenum),str(statenum+1),let,'<epsilon>'])
        c2.write(arc)
        statenum += 1
    finals.append(statenum)
    statenum += 1

#make final states
for final in finals:
    c2.write(str(final))

lm = c2.compile()

#compose into full model
fm = fst.compose(am,lm.arcsort())

#make input wfst
c3 = fst.Compiler(
    isymbols=st,
    osymbols=st,
    keep_isymbols=True,
    keep_osymbols=True
)
c3.write('0 1 K K')
c3.write('1 2 A A')
c3.write('2 3 T T')
c3.write('3 4 A A')
c3.write('4')
inp = c3.compile()

#compose input with full model and print shortest path
print(fst.shortestpath(fst.compose(inp,fm)))

