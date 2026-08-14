from pyfoma import FST

filename = 'past.txt'

f = open(filename,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

fstsg = FST.re("$^rewrite(n:m / _ '-' .* 1 ; 'SG')")

fstpl = FST.re("$^rewrite(n:(im) / _ '-' .* 1 ; 'PL')")

sndsg = FST.re("$^rewrite(n:i / _ '-' .* 2 ; 'SG')")

sndpl = FST.re("$^rewrite(n:(in) / _ '-' .* 2 ; 'PL')")

thrdsg = FST.re("$^rewrite(n:'' / _ '-' .* 3 ; 'SG')")

drop = FST.re("$^rewrite(('-'.*):'',longest=True)")

pmark = FST.re(
	"$fstsg @ $fstpl @ $sndsg @ $sndpl @ $thrdsg",
	{'fstsg':fstsg,'fstpl':fstpl,'sndsg':sndsg,
	'sndpl':sndpl,'thrdsg':thrdsg}
)

stm1 = FST.re("$^rewrite((xuandn):(xundn))")
stm2 = FST.re("$^rewrite((ngah):(niga))")
stm3 = FST.re("$^rewrite((manstn):(munstn))")
stm4 = FST.re("$^rewrite((mandn):(mundn))")
stm5 = FST.re("$^rewrite((kuCk):(kuCik))")
stm6 = FST.re("$^rewrite((drkrdn):(dr ' ' krdn))")
stm7 = FST.re("$^rewrite((s'*'br ' ' krdn):(s'*'brn))")
stm8 = FST.re("$^rewrite((tmam):(tmum))")
stm9 = FST.re("$^rewrite((Hmam):(Hmum))")
stm10 = FST.re("$^rewrite((tkan):(tkun))")

everything = FST.re(
	"$stm1 @ $stm2 @ $stm3 @ $stm4 @ $stm5 @ " + \
	"$stm6 @ $stm7 @ $stm8 @ $stm9 @ $stm10 @ " + \
	"$pmark @ $drop",
	{'stm1':stm1,'stm2':stm2,'stm3':stm3,'stm4':stm4,
	'stm5':stm5,'stm6':stm6,'stm7':stm7,'stm8':stm8,
	'stm9':stm9,'stm10':stm10,'pmark':pmark,'drop':drop}
)

for line in lines:
	bits = line.split('\t')
	inp = '-'.join([bits[0],bits[2]])
	res = list(everything.generate(inp))[0]
	if res != bits[1]:
		print(f'{inp}: {bits[1]}/{res}')

