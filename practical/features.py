import panphon as p

f = open('letters.txt','r')
t = f.read()
f.close()

t = t.split('\n')
t = t[:-1]

ft = p.FeatureTable()

for letter in t:
    print(letter,ft.word_fts(letter)[0],'\n')
