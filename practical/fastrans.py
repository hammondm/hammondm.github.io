letters = {
	'ز':'z',	'ئ':'I',	'ڤ':'B',	'م':'m',
	'ق':'q',	'ذ':'z*',	'پ':'p',	'ن':'n',
	'ش':'S',	'ع':'?*',	'ف':'f',	'و':'u',
	'ژ':'Z',	'ط':'t*',	'ی':'i',	'ە':'?',
	'آ':'a:',	'د':'d',	'ک':'k',	'ب':'b',
	'ؤ':'?',	'ث':'s:',	'ح':'H',	'خ':'x',
	'ء':'?',	'گ':'g',	'ك':'k',	'س':'s',
	'ر':'r',	'چ':'C',	'أ':'?',	'ض':'z+',
	'ظ':'z:',	'ى':'e',	'ت':'t',	'ج':'J',
	'ا':'a',	'ص':'s*',	'ل':'l',	'ه':'h',
	'غ':'G'
}

def dotrans(x):
	trans = ''
	for letter in x:
		if letter in letters:
			trans += letters[letter]
		else:
			trans += letter
	return trans

if __name__ == '__main__':
	filename = '../uni/fas'
	f = open(filename,'r')
	t = f.read()
	f.close()
	t = t.split('\n')[1:-1]
	for line in t:
		bits = line.split('\t')
		b0 = dotrans(bits[0])
		b1 = dotrans(bits[1])
		print(f'{b0}\t{b1}\t{bits[2]}')

