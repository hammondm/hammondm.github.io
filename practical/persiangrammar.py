from pyfoma import FST

#V;1;PL;FUT;COL
#V;1;SG;FUT;COL
#V;2;PL;FUT;COL
#V;2;SG;FUT;COL
#V;3;PL;FUT;COL
#V;3;SG;FUT;COL

prs = FST.re("'PRS' ; ('IPFV'|'PRF'|'PROG')")

pst = FST.re("'PST' (; ('IPFV'|'PRF'|'PROG'))?")

subj = FST.re("'SUBJ' ; ('PRS'|'PST')")

tns = FST.re(
	"('FUT'|$prs|$pst|'PFV'|$subj|'IMP')",
	{'prs':prs,'pst':pst,'subj':subj}
)

v1 = FST.re(
	"V ; [123] ; ('SG'|'PL') ; $tns",
	{'tns':tns}
)

v2 = FST.re(
	"$v1 - (V ; [13] .* 'IMP')",
	{'v1':v1}
)

v3 = FST.re(
	"($v2|V ; 'NFIN'|V ; 'PTCP' ; ('PRS'|'PST')) (; 'COL')?",
	{'v2':v2}
)

verb = FST.re(
	"$v3 - (.* 'FUT' ; 'COL')",
	{'v3':v3}
)

wds = verb.words()
for wd in wds:
	print(''.join([x[0] for x in wd[1]]))

