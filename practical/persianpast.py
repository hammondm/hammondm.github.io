from pyfoma import FST

filename = 'past.txt'

#read in colloquial past tense forms
f = open(filename,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#person marking
fstsg = FST.re("$^rewrite(n:m / _ '-' .* 1 ; 'SG')")
fstpl = FST.re("$^rewrite(n:(im) / _ '-' .* 1 ; 'PL')")
sndsg = FST.re("$^rewrite(n:i / _ '-' .* 2 ; 'SG')")
sndpl = FST.re("$^rewrite(n:(in) / _ '-' .* 2 ; 'PL')")
thrdsg = FST.re("$^rewrite(n:'' / _ '-' .* 3 ; 'SG')")

#get rid of tags at the end
drop = FST.re("$^rewrite(('-'.*):'',longest=True)")

#do all person marking
pmark = FST.re(
	"$fstsg @ $fstpl @ $sndsg @ $sndpl @ $thrdsg",
	{'fstsg':fstsg,'fstpl':fstpl,'sndsg':sndsg,
	'sndpl':sndpl,'thrdsg':thrdsg}
)

#exceptions and stem changes
stm1 = FST.re("$^rewrite((xuandn):(xundn))")
stm2 = FST.re("$^rewrite((ngah):(niga))")
stm3 = FST.re("$^rewrite((manstn):(munstn))")
stm4 = FST.re("$^rewrite((mandn):(mundn))")
stm5 = FST.re("$^rewrite((kuCk):(kuCik))")
stm6 = FST.re("$^rewrite((drkrdn):(dr ' ' krdn))")
stm7 = FST.re("$^rewrite((s'*'br ' ' krdn):(s'*'brn))")
stm8 = FST.re("$^rewrite(a:u / (tm|Hm|a':'r) _ m)")
stm9 = FST.re("$^rewrite((baz):(ua) / _ ' ')")
stm10 = FST.re("$^rewrite(a:u / (pnh|tk|mi|zb) _ n)")
stm11 = FST.re("$^rewrite('?':'' / amz '+' a _)")
stm12 = FST.re("$^rewrite(':':u / a _ mdn)")

#do all extras
extras = FST.re(
	"$stm1 @ $stm2 @ $stm3 @ $stm4 @ $stm5 @ " + \
	"$stm6 @ $stm7 @ $stm8 @ $stm9 @ $stm10 @ " + \
	"$stm11 @ $stm12",
	{'stm1':stm1,'stm2':stm2,'stm3':stm3,'stm4':stm4,
	'stm5':stm5,'stm6':stm6,'stm7':stm7,'stm8':stm8,
	'stm9':stm9,'stm10':stm10,'stm11':stm11,
	'stm12':stm12}
)

#put everything together
everything = FST.re(
	"$extras @ $pmark @ $drop",
	{'extras':extras,'pmark':pmark,'drop':drop}
)

#check that we have everything
for line in lines:
	bits = line.split('\t')
	#input: lemma - tags
	inp = '-'.join([bits[0],bits[2]])
	#get output
	res = list(everything.generate(inp))[0]
	#check
	if res != bits[1]:
		#input: target/output
		print(f'{inp}: {bits[1]}/{res}')

