from pyfoma import FST

#read in all letters
f = open('letters.txt','r')
t = f.read()
f.close()
t = t.split('\n')[:-1]

#FSA for 1,2,3 letters
let1 = FST.re('(' + '|'.join(t) + ')')
let2 = FST.re('$let1 $let1',{'let1':let1})
let3 = FST.re('$let1 $let1 $let1',{'let1':let1})

print(f'1 letter: {let1.pathcount()}')
print(f'2 letters: {let2.pathcount()}')
print(f'3 letters: {let3.pathcount()}')

#read in all vowels
f = open('vowels.txt','r')
t = f.read()
f.close()
t = t.split('\n')[:-1]

#FSA for a single vowel
v = FST.re('(' + '|'.join(t) + ')')

#FSA for word that contains at least 1 vowel
xvx = FST.re('$let1* $v $let1*',{'let1':let1,'v':v})

#1,2,3-segment words with at least 1 vowel
v1 = FST.re('$xvx & $let1',{'xvx':xvx,'let1':let1})
v2 = FST.re('$xvx & $let2',{'xvx':xvx,'let2':let2})
v3 = FST.re('$xvx & $let3',{'xvx':xvx,'let3':let3})

print(f'1 vowel, 1 letter: {v1.pathcount()}')
print(f'1 vowel, 2 letters: {v2.pathcount()}')
print(f'1 vowel, 3 letters: {v3.pathcount()}')

#FSA prohibiting VVV
vvv = FST.re('~(.* $v $v $v .*)',{'v':v})

#FSAs for must have vowel, no VVV, 1,2,3 letters
vvv1 = FST.re(
	'$xvx & $vvv & $let1',
	{'xvx':xvx,'vvv':vvv,'let1':let1}
)
vvv2 = FST.re(
	'$xvx & $vvv & $let2',
	{'xvx':xvx,'vvv':vvv,'let2':let2}
)
vvv3 = FST.re(
	'$xvx & $vvv & $let3',
	{'xvx':xvx,'vvv':vvv,'let3':let3}
)

print(f'1 vowel, 1 letter, vvv: {vvv1.pathcount()}')
print(f'1 vowel, 2 letters, vvv: {vvv2.pathcount()}')
print(f'1 vowel, 3 letters, vvv: {vvv3.pathcount()}')

#FSA prohibiting voiced before voiceless stop
cc = FST.re("~(.*[bdg][ptk].*)")

#putting them all together
cc1 = FST.re('$cc & $vvv1',{'cc':cc,'vvv1':vvv1})
cc2 = FST.re('$cc & $vvv2',{'cc':cc,'vvv2':vvv2})
cc3 = FST.re('$cc & $vvv3',{'cc':cc,'vvv3':vvv3})

print(f'1 vowel, 1 letter, vvv, cc: {cc1.pathcount()}')
print(f'1 vowel, 2 letters, vvv, cc: {cc2.pathcount()}')
print(f'1 vowel, 3 letters, vvv, cc: {cc3.pathcount()}')

