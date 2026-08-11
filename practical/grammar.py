from pyfoma import FST

#indicative tenses
tense = FST.re("'IND' ; ('IPFV'|'PRS'|'PST')")

#imperative
imp = FST.re("2 ; ('SG'|'PL') ; 'IMP'")

#colloquial forms (20)
col = FST.re(
	"'COL' ; (([123] ; ('SG'|'PL') ; $tense)|$imp)",
	{'tense':tense,'imp':imp}
)

#[123] person options
notfour = FST.re("[123] ; ('SG'|'PL')")

#4th person tenses
four = FST.re(
	"(4|$notfour) ; ('IMP'|'SBJV'|('IND' ;" + \
	" ('IPFV'|'PRS'|'PST' (; 'PFV')?)))",
	{'notfour':notfour}
)

lit = FST.re(
	"('LIT' ; $four) - ('LIT' ; 1 ; 'SG' ; 'IMP')",
	{'four':four}
)

verb = FST.re(
	"V ; ($lit|$col|'V.MSDR'|'V.PTCP')",
	{'lit':lit,'col':col}
)

wds = verb.words()
for wd in wds:
	print(''.join([x[0] for x in wd[1]]))

