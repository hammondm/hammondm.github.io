import os,re,sys
from pydub import AudioSegment
from pydub.playback import play
 
#location of textgrids
dirname = 'mfaaligned2/'
#location of wav files
wavdir = 'mfawelsh2/'

#sound is given on command-line
if len(sys.argv) != 2:
	print('usage: python playsounds.py sound')
	exit()
sound = sys.argv[1]

#get names of all the textgrids
filenames = os.listdir(dirname)

#list to store sounds
results = []

#go through all the textgrids
for filename in filenames:
	f = open(dirname+filename,'r')
	t = f.read()
	f.close()
	t = t.split('\n')
	t = t[:-1]
	i = 0
	while i < len(t):
		#assume individual sounds on tier 2
		if 'item [2]:' not in t[i]:
			i += 1
		else:
			i += 6
			break
	while i < len(t):
		xmin = re.sub('.*= ','',t[i+1])
		xmin = float(xmin)
		xmax = re.sub('.*= ','',t[i+2])
		xmax = float(xmax)
		seg = re.sub('.*= "','',t[i+3])
		seg = re.sub('"','',seg)
		seg = re.sub('"','',seg)
		seg = re.sub(' ','',seg)
		results.append((seg,filename,xmin,xmax))
		i += 4

#get matching sounds
matches = []
for result in results:
	if result[0] == sound:
		matches.append(result)

prompt = f'''There are {len(matches)} matches. Responses:
	q		quit
	p		play current item
	n		go to next item
	g#		go to item # (for example: "g35")
'''

#loop over matches
idx = 0
while idx < len(matches):
	print(f'current item: {idx}')
	answer = input(prompt)
	if answer == 'q':
		break
	elif answer == 'n':
		idx += 1
	elif answer == 'p':
		print(f'curent item: {matches[idx]}')
		filename = matches[idx][1]
		filename = filename[:-8] + 'wav'
		start = int(matches[idx][2]*1000) - 30
		print(matches[idx][2],start)
		end = int(matches[idx][3]*1000) + 30
		print(matches[idx][3],end)
		song = AudioSegment.from_wav(wavdir+filename)
		print(f'Playing {filename} from {start} to {end}')
		play(song[start:end])
	elif answer[0] == 'g':
		idx = int(answer[1:])

